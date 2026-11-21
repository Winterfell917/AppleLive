"""Prepare review plots, then export manually approved new_datasets sequences.

python prepare_new_datasets.py
Edit review.json: set start_s/end_s and approved=true for reviewed sequences.
python prepare_new_datasets.py --export
Times are seconds after session-info.startedAtUnixNs. Raw files are never edited.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ACTIONS = ['WalkingStraight', 'WavingLeft', 'WavingRight', 'BothArmRaises',
           'ArmsCrossingAcrossTorso', 'BoxingAlternateArms', 'LeftLegRaises',
           'RightLegRaises', 'Squats', 'JumpingJacks']
DEVICES = ['apple_watch', 'iphone']
APPLE_LH_TO_RH = np.diag([1.0, 1.0, -1.0])

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def load(folder):
    info = read(folder / 'session-info.json')
    origin = int(info['startedAtUnixNs'])
    streams = {}
    for device in DEVICES:
        files = list(folder.glob(device + '-events-*.ndjson'))
        if len(files) != 1:
            raise ValueError(f'{folder}: expected one {device} event file')
        motion, uwb = [], []
        for line in files[0].read_text(encoding='utf-8').splitlines():
            e = json.loads(line)
            if e.get('sessionID') != info['sessionID']:
                raise ValueError('Mixed session IDs: ' + str(folder))
            v = e.get('values', {})
            t = (int(e['timestampUnixNs']) - origin) / 1e9
            if e['sensor'] == 'device_motion':
                r_left = np.array([v[f'rotation_m{i}{j}'] for i in range(1,4) for j in range(1,4)]).reshape(3,3).T
                a_left = np.array([v[f'user_accel_{x}_g'] for x in 'xyz']) * 9.80665
                # Match sensor_apple.sensor: reflect Apple S/I Z once at the
                # ingestion boundary so every downstream rotation is proper
                # and uses MobilePoser's right-handed convention.
                r = APPLE_LH_TO_RH @ r_left @ APPLE_LH_TO_RH
                a = APPLE_LH_TO_RH @ a_left
                motion.append((t, a, r, v.get('attitude_reference_frame')))
            elif e['sensor'] == 'uwb_ranging' and np.isfinite(v.get('distance_m', np.nan)) and v['distance_m'] > 0:
                uwb.append((t, v['distance_m']))
        motion.sort(key=lambda x:x[0])
        if len(motion) < 10:
            raise ValueError('Insufficient motion: ' + str(folder))
        t = np.array([x[0] for x in motion])
        if np.any(np.diff(t) <= 0):
            raise ValueError('Duplicate/reversed motion timestamps: ' + str(folder))
        streams[device] = dict(t=t, a=np.stack([x[1] for x in motion]),
                               r=np.stack([x[2] for x in motion]),
                               refs=sorted(set(str(x[3]) for x in motion)),
                               uwb=np.array(sorted(uwb)).reshape(-1,2))
    return info, streams

def stable(s):
    candidates = []
    for start in np.arange(s['t'][0]+0.5, s['t'][-1]-2.5, 0.25):
        ix = (s['t'] >= start) & (s['t'] < start+2)
        if ix.sum() < 40: continue
        rs = Rotation.from_matrix(s['r'][ix])
        mean = rs.mean()
        angle = np.degrees((mean.inv()*rs).magnitude()).max()
        rms = np.sqrt(np.mean(np.sum(s['a'][ix]**2, axis=1)))
        candidates.append((angle+rms*10, mean.as_matrix(), dict(start_s=float(start), end_s=float(start+2), max_angle_deg=float(angle), accel_rms=float(rms))))
    if not candidates: raise ValueError('No two-second calibration window')
    _, r, q = min(candidates, key=lambda x:x[0])
    q['passed'] = q['max_angle_deg'] <= 12 and q['accel_rms'] <= 1.5
    return r, q

def calibrate(wear):
    # Native CoreMotion S: +Y device head, +Z screen outward.
    # Back upward => screen downward. M: +Y up, +Z forward, +X right.
    # Columns are the right-handed sensor axes expressed in M when the device
    # lies back-up, head (+S y) toward +M z: Sx=-Mx, Sy=+Mz, Sz=+My.
    known = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=float)
    if not np.allclose(known.T @ known, np.eye(3), atol=1e-6) or np.linalg.det(known) < 0.999:
        raise RuntimeError('Known 0rmi placement is not a proper rotation')
    _, flat = load(wear/'0rmi'); _, pose = load(wear/'0rsb')
    result = {}
    for device in DEVICES:
        r0,q0 = stable(flat[device]); rn,qn = stable(pose[device])
        slot = wear.name.split('_')[0 if device=='apple_watch' else 1]
        if slot in ('lw','lh'): bone=np.array([[0,1,0],[-1,0,0],[0,0,1]])
        elif slot in ('rw','rh'): bone=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        else: bone=np.eye(3)
        mi=known@r0.T
        result[device]=dict(R_MI=mi.tolist(), R_SB=((mi@rn).T@bone).tolist(), flat=q0, npose=qn,
                            refs=flat[device]['refs'], passed=q0['passed'] and qn['passed'])
    return result

def plot(streams, entry, target):
    fig, axes = plt.subplots(2,1,figsize=(14,7),sharex=True)
    for ax,device in zip(axes,DEVICES):
        s=streams[device]
        for j,label in enumerate('xyz'): ax.plot(s['t'],s['a'][:,j],lw=.7,label=label)
        ax.plot(s['t'],np.linalg.norm(s['a'],axis=1),color='black',lw=.8,label='norm')
        ax.axvspan(entry['start_s'],entry['end_s'],color='green',alpha=.12)
        ax.set_ylabel(device+'\na_S (m/s²)'); ax.legend(loc='upper right'); ax.grid(alpha=.2)
    axes[-1].set_xlabel('Seconds after session start (both devices share this time axis)')
    axes[-1].xaxis.set_major_locator(MultipleLocator(0.5))
    axes[-1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[-1].tick_params(axis='x', labelrotation=90, labelsize=7)
    fig.suptitle(entry['key']+' — green: proposed/manual action interval')
    fig.tight_layout(); target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,dpi=140); plt.close(fig)

def export(streams, entry, cal, out, fps):
    import torch
    if not entry['approved']: return 'awaiting_review'
    if not all(x['passed'] for x in cal.values()): return 'calibration_failed'
    start,end=entry['start_s'],entry['end_s']
    lo=max(s['t'][0] for s in streams.values()); hi=min(s['t'][-1] for s in streams.values())
    if not lo<=start<end<=hi: raise ValueError('Review interval outside shared device coverage')
    t=np.arange(start,end,1/fps); rs=[]; acc=[]; valid=[]; distances=[]; uv=[]
    for device in DEVICES:
        s=streams[device]; mi=np.array(cal[device]['R_MI'])
        r=Slerp(s['t'],Rotation.from_matrix(s['r']))(t).as_matrix()
        a=np.stack([np.interp(t,s['t'],s['a'][:,j]) for j in range(3)],axis=1)
        idx=np.clip(np.searchsorted(s['t'],t),1,len(s['t'])-1)
        valid.append((s['t'][idx]-s['t'][idx-1])<=.1)
        rs.append(mi@r); acc.append(np.einsum('nij,nj->ni',mi@r,a))
        u=s['uwb']; d=np.full(len(t),np.nan); mask=np.zeros(len(t),bool)
        if len(u):
            ix=np.searchsorted(u[:,0],t,side='right')-1; safe=np.maximum(ix,0)
            mask=(ix>=0)&((t-u[safe,0])<=.5); d[mask]=u[safe[mask],1]
        distances.append(d); uv.append(mask)
    out.mkdir(parents=True,exist_ok=True)
    payload=dict(devices=DEVICES, slots=entry['wear'].split('_'), fps=fps,
                 time_s=torch.tensor(t-start), orientation=torch.tensor(np.stack(rs,axis=1),dtype=torch.float32),
                 acceleration=torch.tensor(np.stack(acc,axis=1),dtype=torch.float32),
                 motion_valid=torch.tensor(np.stack(valid,axis=1)), uwb_distance_m=torch.tensor(np.stack(distances,axis=1)),
                 uwb_valid=torch.tensor(np.stack(uv,axis=1)), calibration=cal)
    torch.save(payload,out/'labels.pt'); write(out/'meta.json',entry)
    write(out/'quality.json',dict(motion_valid_fraction=np.mean(valid,axis=1).tolist(),uwb_valid_fraction=np.mean(uv,axis=1).tolist()))
    return 'exported'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'data/new_datasets')
    parser.add_argument('--export',action='store_true'); parser.add_argument('--fps',type=float,default=30)
    args=parser.parse_args()
    if args.fps<=0: parser.error('fps must be positive')
    review_path=args.root/'review.json'; review=read(review_path) if review_path.exists() else {}
    report=[]
    for wear in sorted((args.root/'raw').glob('*/*')):
        if not wear.is_dir(): continue
        sessions=[(read(p/'session-info.json'),p) for p in wear.iterdir() if p.is_dir() and p.name not in ('0rmi','0rsb')]
        sessions.sort(key=lambda x:x[0]['sessionFolderName'])
        if len(sessions)!=30: raise ValueError(f'{wear}: expected 30 action sessions, got {len(sessions)}; refusing shifted labels')
        cal=calibrate(wear); write(args.root/'calibration'/wear.parent.name/wear.name/'calibration.json',cal)
        for i,(info,folder) in enumerate(sessions):
            key=f'{wear.parent.name}/{wear.name}/{ACTIONS[i//3]}/g{i%3+1:02d}'
            _,streams=load(folder)
            lo=max(s['t'][0] for s in streams.values()); hi=min(s['t'][-1] for s in streams.values())
            if key not in review:
                grid=np.arange(lo,hi,.02); activity=np.zeros(len(grid))
                for s in streams.values(): activity=np.maximum(activity,np.interp(grid,s['t'],np.linalg.norm(s['a'],axis=1)))
                smooth=uniform_filter1d(activity,size=25)
                active=np.flatnonzero(smooth>max(.25,float(np.percentile(smooth,65))*.3))
                start=max(lo,float(grid[active[0]])-.3) if len(active) else lo
                end=min(hi,float(grid[active[-1]])+.3) if len(active) else hi
                review[key]=dict(key=key,subject=wear.parent.name,wear=wear.name,action=ACTIONS[i//3],group=i%3+1,
                                 session_id=info['sessionID'],raw_folder=str(folder.resolve()),start_s=start,end_s=end,
                                 approved=False,boundary_method='activity_candidate_requires_manual_review',
                                 synchronization='device Unix timestamps; cross-device offset not measured')
            entry=review[key]
            if entry['session_id']!=info['sessionID']: raise ValueError('Mapping changed for '+key)
            plot(streams,entry,args.root/'review_plots'/Path(key+'.png'))
            status=export(streams,entry,cal,args.root/'processed'/key,args.fps) if args.export else 'review_ready'
            report.append(dict(key=key,status=status))
            print(key,status,flush=True)
    write(review_path,review); write(args.root/'processing_report.json',report)

if __name__=='__main__': main()

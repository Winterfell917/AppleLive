import socket
import torch
import numpy as np
import time
import threading
import articulate as art
import os
import cv2

try:
    import winsound
except ImportError:
    winsound = None


def beep(frequency=440, duration=600):
    if winsound is not None:
        winsound.Beep(frequency, duration)
    else:
        print("\a", end="", flush=True)


np.set_printoptions(precision=10, suppress=True)
torch.set_printoptions(sci_mode=False)
class DataReceiver(threading.Thread):
    def __init__(self, sock, huawei_sensor, buffer_size=1024):
        super().__init__()
        self.sock = sock
        self.huawei_sensor = huawei_sensor
        self.buffer_size = buffer_size
        self.running = True

    def run(self):
        """持续接收数据并更新缓冲区"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.buffer_size)
                # 调用 HuaweiSensor 的 process_data 方法处理接收到的数据
                msg = data.decode()
                self.huawei_sensor.process_data(msg)
            except Exception as e:
                print("Parse error:", e)
                os._exit(0)

    def stop(self):
        """停止接收线程"""
        self.running = False

class HuaweiSensor:
    def __init__(self, ip="0.0.0.0", port=8989, buffer_size=1024, device_ids=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        
        # 初始化设备ID
        self.device_ids = device_ids or {}
        self.ids = [id for id in device_ids.values()]

        # 初始化数据缓冲区
        self.raw_acc_buffer   = {id: np.zeros((buffer_size, 3)) for id in self.device_ids.values()}
        self.raw_ori_buffer   = {id: np.array([[1, 0, 0, 0]] * buffer_size) for id in self.device_ids.values()}
        self.timestamp_buffer = {id: np.zeros((buffer_size, 1)) for id in self.device_ids.values()}
        
        # 其他模态的数据
        self.gyro_buffer      = {id: np.zeros((buffer_size, 3)) for id in self.device_ids.values()}
        self.mag_buffer       = {id: np.zeros((buffer_size, 3)) for id in self.device_ids.values()}
        self.pressure_buffer  = {id: np.zeros((buffer_size, 1)) for id in self.device_ids.values()} 
        self.ppg_buffer       = {id: np.zeros((buffer_size, 1)) for id in self.device_ids.values()}

        self.time_offsets = [-1, -1, -1, -1, -1, -1, -1]
        
        # 初始化数据接收线程
        self.receiver = DataReceiver(self.sock, self, buffer_size)
        self.receiver.start()
        # # 计算acc bias
        # self.acc_bias = np.zeros((len(self.device_ids), 3))
        # self.cal_acc_bias()

    def get(self, device_id):
        """获取指定设备最新的加速度和四元数数据"""
        acc_data, quat_data, gyro_data, mag_data, pressure_data, ppg_data, timestamp = self.get_latest_data(device_id)
        
        RIS = art.math.quaternion_to_rotation_matrix(torch.tensor(quat_data).float())  # [N, 3, 3]
        aS = acc_data
        aI = RIS.matmul(torch.tensor(aS, dtype=torch.float32).unsqueeze(-1)).squeeze(-1)
        
        # aI[:, 2] -= 9.8
        
        # convert timestamp to tensor
        # * 小心精度损失
        timestamp = timestamp / 1000.0  # 转换为秒
        cur_timestamp = torch.tensor(timestamp)

        return cur_timestamp, aS, aI, RIS, gyro_data, mag_data, pressure_data, ppg_data

    def get_latest_data(self, device_id):
        """获取指定设备的最新加速度和四元数数据"""
        latest_acc = [self.raw_acc_buffer[id][self.time_offsets[id]] for id in device_id]
        latest_ori = [self.raw_ori_buffer[id][self.time_offsets[id]] for id in device_id]
        latest_t   = [self.timestamp_buffer[id][self.time_offsets[id]] for id in device_id]
        
        latest_gyro = [self.gyro_buffer[id][self.time_offsets[id]] for id in device_id]
        latest_mag = [self.mag_buffer[id][self.time_offsets[id]] for id in device_id]
        latest_pressure = [self.pressure_buffer[id][self.time_offsets[id]] for id in device_id]
        latest_ppg = [self.ppg_buffer[id][self.time_offsets[id]] for id in device_id]
        
        latest_acc = np.array(latest_acc)
        latest_ori = np.array(latest_ori)
        latest_t   = np.array(latest_t)
        
        latest_gyro = np.array(latest_gyro)
        latest_mag = np.array(latest_mag)
        latest_pressure = np.array(latest_pressure)
        latest_ppg = np.array(latest_ppg)

        return latest_acc, latest_ori, latest_gyro, latest_mag, latest_pressure, latest_ppg, latest_t

    def process_data(self, msg):
        """接收并处理传感器数据"""
        msg = msg.strip().replace('$', '')
        try:
            parts = msg.split('#')
            dev_id = int(parts[0])
            data_type = parts[1]
            timestamp = int(parts[2])
            values = [float(x) for x in parts[3].split()]
        except Exception as e:
            print(e, msg)
            os._exit(0)
            return
        
        if data_type == 'raw_acceleration' or data_type == 'raw_acceleration_l':
            curr_acc = np.array(values[0:3]).reshape(1, 3)
            self.raw_acc_buffer[dev_id] = np.concatenate([self.raw_acc_buffer[dev_id][1:], curr_acc])
        
        elif data_type == 'raw_angular_velocity' or data_type == 'raw_angular_velocity_l':
            curr_gyro = np.array(values[0:3]).reshape(1, 3)
            self.gyro_buffer[dev_id] = np.concatenate([self.gyro_buffer[dev_id][1:], curr_gyro])
            
        elif data_type == 'raw_magnetic_filed':
            curr_mag = np.array(values[0:3]).reshape(1, 3)
            self.mag_buffer[dev_id] = np.concatenate([self.mag_buffer[dev_id][1:], curr_mag])
            
        elif data_type == 'pressure':
            curr_pressure = np.array(values[0:1]).reshape(1, 1)
            self.pressure_buffer[dev_id] = np.concatenate([self.pressure_buffer[dev_id][1:], curr_pressure])
        
        elif data_type == 'ppg':
            curr_ppg = np.array(values[0:1]).reshape(1, 1)
            self.ppg_buffer[dev_id] = np.concatenate([self.ppg_buffer[dev_id][1:], curr_ppg])

        elif data_type == 'orientation' or data_type == 'orientation_left':
            curr_ori = np.array(values[0:4]).reshape(1, 4)
            self.raw_ori_buffer[dev_id] = np.concatenate([self.raw_ori_buffer[dev_id][1:], curr_ori])
            self.timestamp_buffer[dev_id] = np.concatenate([self.timestamp_buffer[dev_id][1:], np.array([[timestamp]])])
        # 对于其他数据类型，可以继续添加处理逻辑
        # TODO: 更新时间戳
        # self.timestamp_buffer[dev_id] = np.concatenate([self.timestamp_buffer[dev_id][1:], np.array([[timestamp]])])

class CalibratedHuaweiSensor(HuaweiSensor):
    _RMB_Npose = torch.tensor([[[0, 1, 0],  [-1, 0, 0], [0, 0, 1]],         # left wrist
                               [[0, -1, 0], [1, 0, 0], [0, 0, 1]],         # right wrist
                               [[1, 0, 0],  [0, 1, 0],  [0, 0, 1]],          # left thigh
                               [[1, 0, 0],  [0, 1, 0],  [0, 0, 1]],          # right thigh
                               [[1, 0, 0],  [0, 1, 0],  [0, 0, 1]],          # head
                               [[1, 0, 0],  [0, 1, 0],  [0, 0, 1]],          # left foot
                               [[1, 0, 0],  [0, 1, 0],  [0, 0, 1]]]).float() # right foot
    
    # _RMB_Npose = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],            # right thigh
    #                            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],           # left wrist
    #                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float()   # head
    
    def __init__(self, device_ids, buffer_size=1024):
        super().__init__(device_ids=device_ids, buffer_size=buffer_size)
        self.ids = [id for id in device_ids.values()]
        self.N = len(self.ids)
        self.RMI = torch.eye(3).repeat(self.N, 1, 1)
        self.RSB = torch.eye(3).repeat(self.N, 1, 1)
        self.Npose = self._RMB_Npose[self.ids]
        
        # acc_bias
        self.acc_bias = torch.zeros(self.N, 3)

    def get_cali_matrices(self):
        return self.RMI.clone(), self.RSB.clone(), self.acc_bias.clone()
    
    def cal_acc_bias(self):
        print("Calculating accelerometer bias...")
        acc_samples = []
        start_time = time.time()
        
        while time.time() - start_time < 3:  # 收集3秒钟的数据
            _, _, aI, _, _, _, _, _ = super().get(self.ids) # aI: [N, 3]
            acc_samples.append(aI.cpu().numpy())
            
            time.sleep(0.01)  # 每10ms采样一次
        
        if len(acc_samples) > 0:
            acc_samples = np.array(acc_samples)  # 转换为NumPy数组
            self.acc_bias = torch.tensor(acc_samples.mean(axis=0)).float()  # 计算每个设备的平均加速度作为偏置
            print("Accelerometer bias calculated:", self.acc_bias)
        else:
            print("No accelerometer data collected for bias calculation.")
    
    def get(self):
        t, aS, aI, RIS, gyro, mag, pressure, ppg = super().get(self.ids) # aS: [N, 3], RIS: [N, 3, 3] 

        aI = aI - self.acc_bias.to(aI.device)
    
        RMB = self.RMI.matmul(RIS).matmul(self.RSB)
        aM = self.RMI.matmul(aI.unsqueeze(-1)).squeeze(-1)
        
        return t, aS, aI, aM, RIS, RMB, gyro, mag, pressure, ppg

    @staticmethod
    def _normalize_tensor(x: torch.Tensor, dim=-1):
        norm = x.norm(dim=dim, keepdim=True)
        normalized_x = x / norm
        return normalized_x

    @staticmethod
    def _mean_rotation(R0, R1):
        R_avg = (R0 + R1) / 2
        U, S, V = torch.svd(R_avg)
        R = torch.matmul(U, V.transpose(-2, -1)).view(-1, 3, 3)
        m = R.det() < 0
        R[m] = U[m].matmul(torch.diag(torch.tensor([1, 1, -1.], device=R.device))).bmm(V[m].transpose(1, 2))
        return R
    @staticmethod
    def _rotation_matrix_to_axis_angle(r):
        result = [cv2.Rodrigues(_)[0] for _ in r.view(-1, 3, 3).numpy()]
        result = torch.from_numpy(np.stack(result)).float().squeeze(-1)
        return result
    @staticmethod
    def _input(s, skip_input):
        if skip_input:
            print(s)
        else:
            return input(s)

    def _walking_calibration(self, RMI_method, skip_input=False):
        self._input('Huawei: Stand in N pose for 3 seconds, then step forward and stop in the N-pose again.', skip_input)
        time.sleep(2)
        self.cal_acc_bias()
        RIS_N0 = self.get()[4]

        beep()
        print('Step forward now.')
        begin_t = last_t = self.get()[0][0]
        p, v = torch.zeros(self.N, 3), torch.zeros(self.N, 3)
        Cov_pv = Cov_vv = 0
        
        while last_t - begin_t < 3:
            t, _, aI, _, _, _, _, _, _, _ = self.get()
            t = t[0][0]

            if t != last_t:
                dt = 0.01
                last_t = t
                p += dt * v + 0.5 * dt * dt * aI
                v += dt * aI
                Cov_pv += dt * Cov_vv
                Cov_vv += 1
        p_filtered = p - Cov_pv / Cov_vv * v
        RIS_N1 = self.get()[4]

        if RMI_method == '9dof':
            zI = p_filtered.mean(dim=0).repeat(self.N, 1)
        elif RMI_method == '6dof':
            zI = p_filtered
        yI = torch.tensor([0, 0, 1.]).expand(self.N, 3)
        xI = self._normalize_tensor(yI.cross(zI, dim=-1))
        zI = self._normalize_tensor(xI.cross(yI, dim=-1))
        RMI = torch.stack([xI, yI, zI], dim=-2)
        
        RSB0 = RMI.matmul(RIS_N0).transpose(1, 2).matmul(self.Npose)
        RSB1 = RMI.matmul(RIS_N1).transpose(1, 2).matmul(self.Npose)
        
        RSB = self._mean_rotation(RSB0, RSB1)

        err_vertical = p_filtered[:, -1].abs()
        err_RSB = self._rotation_matrix_to_axis_angle(RSB0.bmm(RSB1.transpose(1, 2))).norm(dim=-1) * (180 / np.pi)
        # change err_vertical from 0.1 to 0.4
        if all(err_vertical < 1.0) and all(err_RSB < 80) or skip_input:
            c = 'n'
            print('Calibration succeed: vertical error %s m, RSB error %s deg' % (err_vertical, err_RSB))
        else:
            c = input('Calibration fail: vertical error %s m, RSB error %s deg. Try again? [y]/n' % (err_vertical, err_RSB))
        if c != 'n':
            self._walking_calibration(RMI_method, skip_input)
        else:
            self.RMI = RMI
            self.RSB = RSB

    def calibrate(self, method, skip_input=False):
        r"""
        Calibrate the IMU set.

        :param method: Calibration method. Select from:
            - 'tpose_9dof':     Full T-pose calibration for 9-dof IMU. Two steps: align imu 1 and stand in T-pose.
            - 'tpose_6dof':     Full T-pose calibration for 6-dof IMU. Two steps: align all imus and stand in T-pose.
            - 'tpose_onlyRSB':  T-pose calibration only for sensor-to-bone offset. One step: stand in T-pose.
            - 'npose_9dof':     Full N-pose calibration for 9-dof IMU. Two steps: align imu 1 and stand in N-pose.
            - 'npose_6dof':     Full N-pose calibration for 6-dof IMU. Two steps: align all imus and stand in N-pose.
            - 'npose_onlyRSB':  N-pose calibration only for sensor-to-bone offset. One step: stand in N-pose.
            - 'npose_tpose':    Change-pose calibration for 9-dof IMU. One step: stand in N pose and then change to T-pose.
            - 'walking_9dof':   Walking calibration for 9-dof IMU. One step: stand in N pose, step forward and stop in N pose.
            - 'walking_6dof':   Walking calibration for 6-dof IMU. One step: stand in N pose, step forward and stop in N pose.
        :param skip_input: If true, skip user input and do calibration directly without error check.
        """
        if method == 'tpose_9dof':
            self._fixpose_calibration(RMI_method='9dof', RSB_method='tpose', skip_input=skip_input)
        elif method == 'tpose_6dof':
            self._fixpose_calibration(RMI_method='6dof', RSB_method='tpose', skip_input=skip_input)
        elif method == 'tpose_onlyRSB':
            self._fixpose_calibration(RMI_method='skip', RSB_method='tpose', skip_input=skip_input)
        elif method == 'npose_9dof':
            self._fixpose_calibration(RMI_method='9dof', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_6dof':
            self._fixpose_calibration(RMI_method='6dof', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_onlyRSB':
            self._fixpose_calibration(RMI_method='skip', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_tpose':
            self._changepose_calibration(skip_input=skip_input)
        elif method == 'walking_9dof':
            self._walking_calibration(RMI_method='9dof', skip_input=skip_input)
        elif method == 'walking_6dof':
            self._walking_calibration(RMI_method='6dof', skip_input=skip_input)

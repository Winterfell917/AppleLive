import fs from "node:fs/promises";
import {
  FileBlob,
  PresentationFile,
} from "file:///C:/Users/Lc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const source = String.raw`D:\Lc\studying\Mocap\零初始化日常设备动捕\No.8 Add UWB\report8_260928.pptx`;
const buildDir = String.raw`D:\Lc\studying\Mocap\AppleLive\MocapLive\.codex-ppt-build\report9`;
const draftPath = `${buildDir}\\draft_report9_261012.pptx`;
await fs.mkdir(buildDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(source));

const keep = [
  "sl/y90nupkv",
  "sl/hwbqtkby",
  "sl/ofy9wn61",
  "sl/m90b6t0r",
  "sl/gnmp4jqx",
  "sl/fu1gfa1s",
  "sl/udsvah03",
  "sl/6lsnupw7",
];
presentation.slides.keep(keep);
presentation.slides.reorder(keep);

const setText = (id, value) => {
  presentation.resolve(id).text = value;
};

// 1. Cover
setText("sh/547294r6", "零初始化日常设备动捕");
setText(
  "sh/k3yl0zql",
  "真机位置分类：自采数据集与采集流程\n汇报人：曹琳东\n2026.10.12",
);

// 2. Project overview
setText("sh/9072xkry", "一、项目简介");
setText(
  "sh/ozy1ofad",
  [
    "目标：在不知道设备佩戴位置、无需 T-Pose 等初始化动作的前提下，实现即穿即用的动作捕捉。",
    "当前问题：使用合成数据训练的位置分类网络，在真机上的效果很差。",
    "本阶段目标：先验证真机数据能否解决手机和手表的身体位置分类。",
    "输入：手机、手表的朝向与加速度时序，以及两设备间的 UWB 距离。",
    "输出：手机和手表所在的身体槽位及分类置信度。",
    "数据路线：本人数据用于训练与验证，后续用其他人或其他设备的数据测试泛化。",
  ].join("\n"),
);

// 3. Last week's work
setText("sh/cza94vmx", "二、上周工作和结论");
setText(
  "sh/d0jax03i",
  [
    "学长建议：合成集到真机的域差异过大，应先采一个小规模真机数据集训练位置分类网络。",
    "- 上周重点：确定自采数据的规模、动作种类、佩戴组合、标签和文件格式。",
    "- 已完成：写好训练数据采集脚本，固化标定、录制、传输、对齐和质量检查协议。",
    "- 已调整：只用 iPhone 标定 R_MI，Watch 不单独标定，直接共享手机的 R_MI。",
    "- 当前阻塞：iPhone 与 Watch 采集软件仍需学长更新，未按原计划采完全部数据。",
    "- 结论：采集方案和代码已基本就绪，下一步取决于采集 App 更新和真机数据落地。",
  ].join("\n"),
);

// 4. Dataset plan table
setText("sh/xc3mho32", "2.1  自采训练集规模");
const table = presentation.resolve("tb/lw7u18n6");
const cells = [
  ["维度", "设置", "数量", "单条时长", "目标频率", "说明"],
  ["动作", "Walking 等 9 类", "9", "20 s", "30 FPS", "覆盖下肢与上肢动作"],
  ["佩戴", "Watch L/R × Phone L/R", "4", "—", "—", "腕部与大腿口袋槽位真值"],
  ["重复", "每个配置 3 组", "3", "—", "—", "降低单次采集偶然性"],
  ["手机朝向", "屏幕朝内/朝外，机头向下", "2", "—", "—", "固定口袋放置约定"],
  ["合计", "9 × 4 × 3 × 2", "216 条", "72 min", "—", "仅统计正式动作时长"],
];
for (let row = 0; row < cells.length; row += 1) {
  for (let col = 0; col < cells[row].length; col += 1) {
    table.cells.set(row, col, cells[row][col]);
  }
}
setText(
  "sh/zelojyl8",
  "每个动作覆盖 4 种佩戴、3 组重复和 2 种手机朝向。目录结构固定，可中断后继续采集。",
);

// 5. Data and labels
setText("sh/cb2tkvap", "2.2  数据内容与标签格式");
setText(
  "sh/dcbud0ra",
  [
    "- 原始文件：iPhone / Watch 的 device_motion、UWB 距离、设备时间戳和 session ID。",
    "- 槽位真值：Watch 为左腕或右腕，Phone 为左大腿或右大腿口袋。",
    "- 标定结果：iPhone 由 N-pose 和向前走得到 R_MI，Watch 直接使用同一个 R_MI。",
    "- 朝向序列：orientation = R_MI @ RIS，统一到模型坐标系。",
    "- 加速度序列：acceleration = R_MI @ RIS @ aS，单位为 m/s²。",
    "- 统一按 30 FPS 对齐并保存有效性掩码，UWB 从原始事件按时间戳对齐。",
  ].join("\n"),
);

// 6. Collection protocol
setText("sh/18byd4zy", "2.3  采集与标定协议");
setText(
  "sh/g72x4zyd",
  "核心约定：手机和手表全程本地记录；只在标定阶段让 iPhone 发送实时 UDP。",
);
setText(
  "sh/ri9g7uhw",
  [
    "- 两台设备开始同一条本地采集，电脑只等待 iPhone 的 device_motion。",
    "- N-pose 先检查静止稳定性，通过后再做向前走。",
    "- iPhone 向前走计算 R_MI；失败立即重做，不开始正式动作。",
    "- 标定通过后关闭手机 UDP，再录制 20 秒正式动作；Watch 始终不实时传输。",
    "- 结束后 Watch 整文件传到手机，电脑导入两份文件并再次校验标定窗口。",
  ].join("\n"),
);

// 7. Implementation progress and blocker
setText("sh/1cfmhgne", "2.4  采集脚本与当前进度");
setText(
  "sh/0b65obm9",
  [
    "- 自动生成 216 条计划，按动作、佩戴、重复组和手机朝向建立目录。",
    "- capture.json 记录动作标签、槽位真值、N-pose、向前走和正式动作窗口。",
    "- 自动识别 iPhone / Watch 文件，完成 30 FPS 对齐并输出 labels、meta 和 quality。",
    "- 已验证：N-pose 失败会立即停止；Watch 与 iPhone 使用完全相同的 R_MI。",
    "- 已验证：模型坐标系朝向、加速度和槽位真值可以正常保存与回读。",
    "- 尚未完成：采集 App 更新前无法稳定跑完整训练集，真实样本数量仍不足。",
  ].join("\n"),
);

// 8. Next week
setText("sh/xc7eds76", "三、下周计划");
setText(
  "sh/wbydknq1",
  [
    "- 1. 等采集 App 更新后，优先完成 216 条本人训练数据，并同步检查文件完整性和标定质量。",
    "- 2. 用自采真机数据训练位置分类网络，在留出的验证集上检查可行性和 UWB 的实际增益。",
    "- 3. 有余力时，用其他参与者或其他设备的数据测试泛化能力。",
  ].join("\n"),
);
setText(
  "sh/jepwf2pc",
  "判断标准：真机训练后，验证集应明显改善手机口袋位置分类；独立测试集用于判断模型能否跨人、跨设备泛化。",
);

// Remove stale speaker notes inherited from the prior report.
const notes = await presentation.inspect({
  select: "$.slides[*].notes",
  include: ["id", "text"],
});
for (const record of notes.records) {
  if (record?.id) presentation.resolve(record.id).text = "";
}

const content = await presentation.inspect({
  select: "$.slides[*]",
  include: ["id", "index", "layout", "title", "elements.id", "elements.kind", "elements.text", "elements.position", "elements.style"],
});
await fs.writeFile(`${buildDir}\\content.ndjson`, content.ndjson, "utf8");

for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(`${buildDir}\\slide-${String(index + 1).padStart(2, "0")}.png`, new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout", detail: "full" });
  await fs.writeFile(`${buildDir}\\slide-${String(index + 1).padStart(2, "0")}.layout.json`, await layout.text(), "utf8");
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(`${buildDir}\\montage.webp`, new Uint8Array(await montage.arrayBuffer()));

const draft = await PresentationFile.exportPptx(presentation);
await draft.save(draftPath);
console.log(draftPath);

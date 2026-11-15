import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  FileBlob,
  PresentationFile,
} from "file:///C:/Users/Lc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const workspaceDir = String.raw`D:\Lc\studying\Mocap\AppleLive\MocapLive`;
const sourceTemplatePath = String.raw`D:\Lc\studying\Mocap\零初始化日常设备动捕\No.8 Add UWB\report8_260928.pptx`;
const draftPath = path.join(workspaceDir, ".codex-ppt-build", "report9", "draft_report9_261012.pptx");
const stagingDir = path.join(workspaceDir, ".codex-finalizer");
const candidatePath = path.join(stagingDir, "candidate_report9_261012_final.pptx");
const finalPath = path.join(workspaceDir, "output", "report9_261012_final.pptx");
const skillDir = String.raw`C:\Users\Lc\.codex\plugins\cache\openai-primary-runtime\presentations\26.1011.11113\skills\presentations`;
const pythonExecutable = String.raw`C:\Users\Lc\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`;
process.env.RUNTIME_NODE_MODULES = String.raw`C:\Users\Lc\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules`;

await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(draftPath));
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const { finalizePresentation } = await import(pathToFileURL(
  path.join(skillDir, "container_tools", "artifact_tool_utils.mjs"),
).href);

const result = await finalizePresentation({
  explicitTotalSlideCount: 8,
  requiredNativeTableOwnerSlides: [4],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable,
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "4",
  ],
  fontPolicy: {
    basis: "reference",
    families: ["黑体", "微软雅黑"],
    referencePath: sourceTemplatePath,
    referenceSha256: "2330d3a956bcb6a60e1b271a852a000f5c1a6af693915ca10cfde4a6aba30224",
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "report9_261012_final.pptx.validation.json"),
});

console.log(JSON.stringify({ finalPath: result.finalPath, receiptPath: result.receiptPath }, null, 2));

import fs from "node:fs/promises";
import {
  FileBlob,
  PresentationFile,
} from "file:///C:/Users/Lc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const source = String.raw`D:\Lc\studying\Mocap\AppleLive\MocapLive\output\report9_261012_final.pptx`;
const outDir = String.raw`D:\Lc\studying\Mocap\AppleLive\MocapLive\.codex-ppt-build\report9-final-preview-v2`;
await fs.mkdir(outDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(`${outDir}\\slide-${String(index + 1).padStart(2, "0")}.png`, new Uint8Array(await png.arrayBuffer()));
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(`${outDir}\\montage.webp`, new Uint8Array(await montage.arrayBuffer()));
console.log(JSON.stringify({ slideCount: presentation.slides.items.length, slideSize: presentation.slideSize }, null, 2));

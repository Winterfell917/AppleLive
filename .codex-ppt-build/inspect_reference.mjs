import fs from "node:fs/promises";
import {
  FileBlob,
  PresentationFile,
} from "file:///C:/Users/Lc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const source = String.raw`D:\Lc\studying\Mocap\零初始化日常设备动捕\No.8 Add UWB\report8_260928.pptx`;
const outDir = String.raw`D:\Lc\studying\Mocap\AppleLive\MocapLive\.codex-ppt-build\reference`;
await fs.mkdir(outDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const inspected = await presentation.inspect({
  select: "$.slides[*]",
  include: [
    "id", "index", "layout", "title",
    "elements.id", "elements.kind", "elements.name", "elements.text",
    "elements.alt", "elements.position", "elements.style", "notes.text",
  ],
});
await fs.writeFile(`${outDir}\\content.ndjson`, inspected.ndjson, "utf8");

const layoutInspect = await presentation.inspect({
  select: "$.slides[*]",
  include: ["id", "index", "layout", "elements.id", "elements.kind", "elements.position", "elements.style"],
});
await fs.writeFile(`${outDir}\\layout.ndjson`, layoutInspect.ndjson, "utf8");

for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(`${outDir}\\slide-${String(index + 1).padStart(2, "0")}.png`, new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout", detail: "full" });
  await fs.writeFile(`${outDir}\\slide-${String(index + 1).padStart(2, "0")}.layout.json`, await layout.text(), "utf8");
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(`${outDir}\\montage.webp`, new Uint8Array(await montage.arrayBuffer()));

console.log(JSON.stringify({
  slideCount: presentation.slides.items.length,
  slideSize: presentation.slideSize,
  masters: presentation.masters.items.map((item) => ({ id: item.id, name: item.name })),
  layouts: presentation.layouts.items.map((item) => ({ id: item.id, name: item.name, placeholders: item.placeholders.summary() })),
}, null, 2));

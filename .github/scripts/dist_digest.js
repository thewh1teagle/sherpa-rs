import { $ } from "bun";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// Define file paths
const filePath = fileURLToPath(import.meta.url);
const dirPath = path.dirname(filePath);
const rootPath = path.join(dirPath, "..", "..");
const distPath = path.join(rootPath, "sys/dist.txt");
const tmpDir = path.join(rootPath, ".tmp");
console.log(process.argv);
const [, , currentTag, newTag] = process.argv;
if (!currentTag || !newTag) {
  console.error("specify tags");
  process.exit(1);
}

if (!fs.existsSync(tmpDir)) {
  fs.mkdirSync(tmpDir);
}

// Read and process the dist.txt file
var distData = fs.readFileSync(distPath, "utf-8");
console.log(`Replace ${currentTag} with ${newTag}`);
distData = distData.replace(new RegExp(currentTag, "g"), newTag);
fs.writeFileSync(distPath, distData);
const lines = distData.split("\n");

for (const line of lines) {
  // Skip commented lines

  if (line.startsWith("#")) continue;

  // Parse the line and extract necessary fields
  const [_features, _arch, url, name, sha256, _dynamic, sizeMB] =
    line.split(/\s+/);

  var text = distData;
  var tmpPath = "";
  try {
    if (url && name && sha256) {
      tmpPath = path.join(tmpDir, name);
      if (!fs.existsSync(tmpPath)) {
        await $`wget -nc -q --show-progress ${url} -O ${tmpPath}`;
      }

      const file = Bun.file(tmpPath);
      const hasher = new Bun.CryptoHasher("sha256");
      const bytes = await file.bytes();
      const newSizeMB = (bytes.length / 1048576).toFixed(0);
      hasher.update(bytes);
      const newSha256 = hasher.digest().toString("hex").toUpperCase();

      console.log("replace ", sha256, newSha256);
      text = text.replace(sha256, newSha256);
      text = text.replace(sizeMB, newSizeMB);
      fs.writeFileSync(distPath, text);
    }
  } catch (e) {
    console.error(e);
    if (tmpPath) {
      fs.rmSync(tmpPath, { force: true });
    }
  }
}
console.log(`Updated ${distPath}`);

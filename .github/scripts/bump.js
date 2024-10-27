import { fileURLToPath } from "bun";
import { readFile, writeFile } from "fs/promises";
import path from "path";
import semver from "semver";

const args = process.argv.slice(2);
const incrementType = args[0];

function inc(version) {
  const v = semver.parse(version);
  if (incrementType) {
    return semver.inc(version, incrementType);
  }
  return semver.prerelease(version)
    ? semver.inc(version, "prerelease")
    : v.patch === 9
    ? v.minor === 9
      ? semver.inc(version, "major")
      : semver.inc(version, "minor")
    : semver.inc(version, "patch");
}

function replaceLineWith(content, { startsWith, value, replaceValue }) {
  return content
    .split("\n")
    .map((line) =>
      startsWith.some((c) => line.includes(c))
        ? line.replace(value, replaceValue)
        : line
    )
    .join("\n");
}

async function bumpCrate(path) {
  var content = await readFile(path);
  var config = Bun.TOML.parse(content);
  var version = config.package.version;
  var content = replaceLineWith(content.toString(), {
    value: version,
    replaceValue: inc(version),
    startsWith: ["sherpa-rs-sys", "version ="],
  });
  await writeFile(path, content);
}

const root = path.resolve(fileURLToPath(import.meta.url), "../../../");

bumpCrate(path.join(root, "Cargo.toml"));
bumpCrate(path.join(root, "sys/Cargo.toml"));

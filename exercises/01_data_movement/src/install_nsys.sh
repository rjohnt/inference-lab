#!/usr/bin/env bash
set -euo pipefail
package=${1:?Usage: install_nsys.sh downloaded-NVIDIA-CLI.deb}
expected=b896cb2b9586ddf617c363a43bababad0a015dff4c77d8f0fbb9c26144056a69
actual=$(sha256sum -- "$package")
if [[ "${actual%% *}" != "$expected" ]]; then
  echo 'Package checksum differs from the reviewed 2026.4.1 download.' >&2
  exit 1
fi
install_root="$HOME/.local/opt/nsight-systems-cli-2026.4.1"
binary="$install_root/opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/nsys"
mkdir -p -- "$install_root" "$HOME/.local/bin"
if [[ -e "$HOME/.local/bin/nsys" && ! -L "$HOME/.local/bin/nsys" ]]; then
  echo 'Refusing to replace an existing regular file at the nsys entry point.' >&2
  exit 1
fi
dpkg-deb --extract "$package" "$install_root"
"$binary" --version
ln -sfn -- "$binary" "$HOME/.local/bin/nsys"
echo 'Installed. Ensure $HOME/.local/bin is first on PATH, then run: nsys --version'

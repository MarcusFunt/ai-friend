#!/usr/bin/env bash
set -euo pipefail

# An interactive helper that checks for Docker and installs it when missing.

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' not found. Please install it and re-run this script." >&2
    exit 1
  fi
}

already_installed() {
  echo "Docker is already installed: $(docker --version)"
}

confirm_install() {
  read -r -p "Docker is not installed. Would you like to install it now? [y/N]: " reply
  case "$reply" in
    [yY][eE][sS]|[yY])
      return 0
      ;;
    *)
      echo "Installation cancelled."
      return 1
      ;;
  esac
}

set_sudo() {
  if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
      SUDO="sudo"
    else
      echo "This script needs root privileges. Please run as root or install sudo." >&2
      exit 1
    fi
  else
    SUDO=""
  fi
}

setup_apt_repo() {
  require_cmd curl
  require_cmd gpg
  require_cmd tee
  require_cmd dpkg
  require_cmd awk

  $SUDO install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${ID}/gpg" | $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  $SUDO chmod a+r /etc/apt/keyrings/docker.gpg

  repo_entry="deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable"
  echo "$repo_entry" | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
}

install_docker_apt() {
  require_cmd apt-get
  $SUDO apt-get update
  $SUDO apt-get install -y ca-certificates curl gnupg
  setup_apt_repo
  $SUDO apt-get update
  $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

main() {
  if command -v docker >/dev/null 2>&1; then
    already_installed
    exit 0
  fi

  if ! confirm_install; then
    exit 1
  fi

  if [ ! -r /etc/os-release ]; then
    echo "/etc/os-release is required to detect your distribution." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  . /etc/os-release

  set_sudo

  case "$ID" in
    ubuntu|debian)
      install_docker_apt
      ;;
    *)
      echo "Automatic installation is only supported on Debian/Ubuntu in this script." >&2
      echo "Please install Docker manually for distribution: $ID" >&2
      exit 1
      ;;
  esac

  echo "Docker installation complete. You may need to log out and back in for group changes to apply."
}

main "$@"

#!/bin/bash

# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2024-2025 Intel Corporation

set -eo pipefail
# This script is used to build the docker image of Intel Video Super Resolution.
#   Usage:  docker_build <PLATFORM> <OS> <OS_VERSION>
#   Example: docker_build xeon ubuntu 22.04 or docker_build flex ubuntu 22.04

# Uncomment if it is intended to not unset no_proxy and NO_PROXY and pass it to the context
# force_no_env_unset=1
# Uncoment or set exact same variable to force not to print logo animation of intel
# force_no_intel_logo=1

SCRIPT_DIR="$(readlink -f "$(dirname -- "${BASH_SOURCE[0]}")")"
. "${SCRIPT_DIR}/scripts/common.sh"

[ -z "${force_no_intel_logo}" ] && print_logo

log_info Starting script execution "${BASH_SOURCE[0]}"
OS="${OS:-ubuntu}"
VERSION="${VERSION:-22.04}"
PLATFORM="${PLATFORM:-xeon}"

if [ -n "$1" ]; then
    case "$(printf %s "$1" | tr '[:upper:]' '[:lower:]')" in
        xeon) PLATFORM="xeon" ;;
        flex) PLATFORM="flex"; OS="ubuntu"; VERSION="22.04" ;;
        *) log_error "Platform $1 is not yet supported. Use: [xeon, flex]"; exit 1
    esac
	shift
fi

if [ -n "$1" ]; then
    case "$(printf %s "$1" | tr '[:upper:]' '[:lower:]')" in
        ubuntu) OS="ubuntu" ;;
        centos) OS="centos"; VERSION="9" ;;
        rocky|rockylinux) OS="rockylinux"; VERSION="9-mini" ;;
        *) log_error "Linux distributtion $1 is not yet supported. Use: [ubuntu, centos, rocky]"; exit 1
    esac
	shift
fi

if [ -n "$1" ]; then
    if [ $OS = "ubuntu" ]; then
        case "$1" in
            18.04|20.04|22.04) VERSION="$1" ;;
            *) log_error "Ubuntu release $1 is not yet supported. Use: [18.04, 20.04, 22.04]"; exit 1
        esac
		DEFAULT_CACHE_REGISTRY="${DEFAULT_CACHE_REGISTRY:-docker.io}"
    fi
    if [ $OS = "centos" ]; then
        VERSION="9"
        case "$1" in
            9|stream9) VERSION="9" ;;
            *) log_error "CentOS release $1 is not yet supported, Use: [stream9]"; exit 1
        esac
		DEFAULT_CACHE_REGISTRY="${DEFAULT_CACHE_REGISTRY:-quay.io}"
    fi
    if [ $OS = "rockylinux" ]; then
        case "$1" in
            9|9-mini) VERSION="9-mini" ;;
            *) log_error "RockyLinux release $1 is not yet supported, Use: [9, 9-mini]"; exit 1
        esac
		DEFAULT_CACHE_REGISTRY="${DEFAULT_CACHE_REGISTRY:-docker.io}"
    fi
	shift
fi

DOCKER_PATH="Xeon"
if [ $PLATFORM = "flex" ]; then
    OS="ubuntu"
    VERSION="22.04"
    DOCKER_PATH="Flex"
fi

if [[ ! $(grep -q "\.intel.com" <<< "${no_proxy}${NO_PROXY}") ]]; then
	if [ -z "${force_no_env_unset}" ]; then
	    log_info "Unsetting no_proxy and NO_PROXY env values for docker buildx build."
	    log_info "Disable this behavior by setting force_no_env_unset env variable"
        log_info "to any non-empty value."
	    log_info "\tExample: force_no_env_unset=1"
	    unset no_proxy
	    unset NO_PROXY
	else
		log_info Non-empty force_no_env_unset flag is set.
        log_info Forcing no-unset behavior.
	fi
fi

# Read proxy variables from env to pass them to the builder
set -x
BUILD_ARGUMENTS=$(compgen -e | sed -nE '/_(proxy|PROXY)$/{s/^/--build-arg /;p}')

IMAGE_TAG="${IMAGE_TAG:-${OS}-${VERSION}}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-docker.io}"
IMAGE_CACHE_REGISTRY="${IMAGE_CACHE_REGISTRY:-${DEFAULT_CACHE_REGISTRY}}"

docker buildx build \
    ${BUILD_ARGUMENTS} \
	-f "${SCRIPT_DIR}/docker/${DOCKER_PATH}/Dockerfile.${OS}${VERSION}" \
	-t "${IMAGE_REGISTRY}/raisr/raisr-${PLATFORM}:${IMAGE_TAG}" \
	--build-arg IMAGE_REGISTRY="${IMAGE_REGISTRY}" \
    --build-arg IMAGE_CACHE_REGISTRY="${IMAGE_CACHE_REGISTRY}" \
	"$@" "${SCRIPT_DIR}"
set +x
log_info Finished script execution "${BASH_SOURCE[0]}"

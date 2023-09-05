#!/bin/sh

# This script is used to build the docker image of Intel Video Super Resolution.

# Usage: docker_build <OS> <OS_VERSION>
# for example: docker_build ubuntu 22.04

OS="ubuntu"
VERSION="22.04"

if [ -n "$1" ]; then
    case "$(printf %s "$1" | tr '[:upper:]' '[:lower:]')" in
	    ubuntu) OS="ubuntu" ;;
	    centos) OS="centos" VERSION="7.9" ;;
	    *) echo "This version of dockerfile does not exist will build defualt image with ubuntu22.04."
    esac
fi

if [ -n "$2" ]; then
	if [ $OS = "ubuntu" ]; then
		case "$2" in
			18.04) VERSION="18.04" ;;
			22.04) VERSION="22.04" ;;
			*) echo "This version of dockerfile does not exist will build defualt image with ubuntu22.04."
		esac
	fi
	if [ $OS = "centos" ]; then
		case "$2" in
			7.9) VERSION="7.9" ;;
			*) echo "This version of dockerfile does not exist will build defualt image with centos7.9."
		esac
	fi
fi

docker build -f ./docker/Dockerfile.${OS}${VERSION} --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -t raisr:${OS}${VERSION} .


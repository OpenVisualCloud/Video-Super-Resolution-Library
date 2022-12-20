/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// API Version
#define RAISR_VERSION_MAJOR (22)
#define RAISR_VERSION_MINOR (12)

#define RAISR_CHECK_VERSION(major, minor)                                \
    (RAISR_VERSION_MAJOR > (major) ||                                    \
    (RAISR_VERSION_MAJOR == (major) && RAISR_VERSION_MINOR > (minor)) || \
    (RAISR_VERSION_MAJOR == (major) && RAISR_VERSION_MINOR == (minor))

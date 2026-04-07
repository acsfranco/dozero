#pragma once

#include "file_ops.h"

typedef struct {
  void *ctx;
  file_ops_t *ops;
} file_t;

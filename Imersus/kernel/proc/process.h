#pragma once

#include "../fs/file_ops.h"
#include "../fs/file.h"

#define MAX_FD 32

typedef struct {
  file_t *fd_table[MAX_FD];
} process_t; ///////////////////////////////////////////// Faltou o ;

#pragma once

typedef struct {
  int (*read)(void *, char *, int);
  int (*write)(void *, char *, int);
} file_ops_t;

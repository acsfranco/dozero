#include "fs.h"
#include "file.h"
#include "../proc/process.h" /////////////////////////// tirei kernel

extern process_t current_process;

int write(int fd, char *buf, int size){
  file_t *file = current_process.fd_table[fd];
  return file->ops->write(file->ctx, buf, size);
}

int read(int fd, char *buf, int size){
  file_t *file = current_process.fd_table[fd];
  return file->ops->read(file->ctx, buf, size);
}

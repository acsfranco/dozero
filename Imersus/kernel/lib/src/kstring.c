int strlen(char *str) {
  int i = 0;

  while(*str) {
    i++;
    str++;
  }

  return i;
}

int strequals(char *str1, char *str2) {
  int equals = 1;
  
  if (strlen(str1) != strlen(str2))
    return 0;

  while (*str1) {
    if (*str1 != *str2)
      return 0;
    str1++;
    str2++;
  }

  return 1;
}

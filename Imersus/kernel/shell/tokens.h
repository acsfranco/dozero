#pragma once
#define MAX_TOKENS 17
#define MAX_TKLEN 100

int gettokens(char *cmdstr, char tokens[MAX_TOKENS][MAX_TKLEN]);

#include<SDL2/SDL.h>
#include<SDL2/SDL_ttf.h>
#include<stdio.h>
#include<stdint.h>

#include "../../redesneurais/src/versao2-backprop/neuron.h"
#include "../../redesneurais/src/versao2-backprop/neuralnet.h"
#include "../../redesneurais/src/versao2-backprop/netmath.h"

SDL_Color white = {255, 255, 255, 255};
SDL_Color black = {0, 0, 0, 255};
SDL_Color yellow = {255, 255, 0, 255};
SDL_Color cyan = {0, 255, 255, 255};
SDL_Color blue = {0, 0, 255, 255};
SDL_Color green = {0, 255, 0, 255};

float input[784];

void drawPainel(SDL_Renderer *renderer) {
  SDL_Rect rect = {39, 159, 282, 282};
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
  SDL_RenderDrawRect(renderer, &rect); 
  SDL_Rect recc = {40, 160, 280, 280};
  SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
  SDL_RenderFillRect(renderer, &recc);
}

void printtext(SDL_Renderer *renderer, uint16_t x, uint16_t y, const char *text, SDL_Color color, uint8_t hcentered, uint8_t vcentered, uint32_t fontsize) {
  TTF_Font *font = TTF_OpenFont("DejaVuSans.ttf", fontsize);
  SDL_Surface *surface = TTF_RenderText_Solid(font, text, color);
  SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
  if (hcentered) {
    x -= surface->w / 2;
  }
  if (vcentered) {
    y -= surface->h / 2;
  }
  SDL_Rect rect = {x, y, surface->w, surface->h};
  SDL_RenderCopy(renderer, texture, NULL, &rect);
  SDL_FreeSurface(surface);
  SDL_DestroyTexture(texture);
  TTF_CloseFont(font);
}

void createbutton(SDL_Renderer *renderer, SDL_Rect rect, SDL_Color bgcolor, SDL_Color txtcolor, const char *text) {
  SDL_SetRenderDrawColor(renderer, bgcolor.r, bgcolor.g, bgcolor.b, 255);
  SDL_RenderFillRect(renderer, &rect);
  printtext(renderer, rect.x + (rect.w / 2), rect.y + (rect.h / 2), text, txtcolor, 1, 1, 36);
}

void hbar(SDL_Renderer *renderer, uint16_t x, uint16_t y, float percent, SDL_Color color) {
  SDL_Rect frectc = {x + 1, y + 1, 298, 30};
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  SDL_RenderFillRect(renderer, &frectc);

  SDL_Rect rect = {x, y, 300, 32};
  SDL_SetRenderDrawColor(renderer, 140, 140, 140, 255);
  SDL_RenderDrawRect(renderer, &rect);

  SDL_Rect frect = {x + 1, y + 1, 298.0f * percent, 30};
  SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 255);
  SDL_RenderFillRect(renderer, &frect);
  char buffer[10];
  sprintf(buffer, "%.0f%%", percent * 100);
  printtext(renderer, x + 150, y + 16, buffer, white, 1, 1, 26);
}

void showresult(SDL_Renderer *renderer, float percents[10]) {
  float highest = percents[0];
  uint8_t hindex = 0;

  for (uint8_t i = 0; i < 10; i++) {
    if (percents[i] > highest) {
      highest = percents[i];
      hindex = i;
    }
  }

  char buffer[10];
  uint16_t k = 160;

  for (uint8_t i = 0; i < 10; i++) {
    sprintf(buffer, "%d -", i);
    printtext(renderer, 400, k, buffer, (hindex == i) ? cyan : yellow, 0, 0, 36);
    k += 40;
  }
  
  k = 167;
  for (uint8_t i = 0; i < 10; i++) {
    hbar(renderer, 460, k, percents[i], blue);
    k += 40;
  }
}

uint8_t insideRegion(SDL_Rect rect, uint16_t x, uint16_t y) {
  return x >= rect.x && y >= rect.y && x <= rect.x + rect.w && y <= rect.y + rect.h;
}

void resetInput() {
  for (uint16_t i = 0; i < 784; i++)
    input[i] = 0;
}

void cleanButtonClickHandle(SDL_Event e, SDL_Renderer *renderer) {
  drawPainel(renderer);
  resetInput();
}

void recognizeButtonClickHandle(SDL_Event e) {
  printf("Em construção!!!\n");
}

void setInput(uint16_t x, uint16_t y) {
  input[y * 28 + x] = 1.0f;
}

void painelMouseMotionHandler(SDL_Renderer *renderer, uint16_t mx, uint16_t my, uint8_t drawing) {
  if (drawing) {
    SDL_Rect rect = {(mx / 10) * 10, (my / 10) * 10, 10, 10};
    SDL_SetRenderDrawColor(renderer, cyan.r, cyan.g, cyan.b, 255);
    SDL_RenderFillRect(renderer, &rect);
    setInput((mx - 40.0f) / 10.0f, (my - 160.0f) / 10);
  }
}

void main() {
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window *win = SDL_CreateWindow(
    "Reconhecimento de Digitos Numéricos Manuscritos",
    SDL_WINDOWPOS_CENTERED,
    SDL_WINDOWPOS_CENTERED,
    800, 600, 0
  );

  TTF_Init();

  SDL_Renderer *renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
  SDL_Event e;
  
  resetInput();
  drawPainel(renderer);
  printtext(renderer, 400, 20, "-- Do Zero --", white, 1, 0, 36);
  printtext(renderer, 400, 80, "Reconhecimento de digitos manuscritos", white, 1, 0, 36);
  printtext(renderer, 180, 130, "Desenhe seu digito aqui", white, 1, 0, 22);
  SDL_Rect rect_c = {39, 460, 282, 60};
  SDL_Rect rect_r = {39, 530, 282, 60};
  createbutton(renderer, rect_c, green, black, "Limpar");
  createbutton(renderer, rect_r, blue, white, "Reconhecer");
  float percents[] = {0.1, 0.3, 0.2, 0.8, 0.4, 0.1, 0.5, 0.4, 0.3, 0.1};
  showresult(renderer, percents);
  
  uint8_t running = 1;
  uint8_t drawing = 0;

  while (running) {
    while (SDL_PollEvent(&e)) {
      switch(e.type) {
        case SDL_QUIT:
          running = 0;
          break;
        case SDL_MOUSEBUTTONDOWN:
          drawing = 1;
          if (insideRegion(rect_c, e.button.x, e.button.y)) {
            cleanButtonClickHandle(e, renderer);
          } else if (insideRegion(rect_r, e.button.x, e.button.y)) {
            recognizeButtonClickHandle(e);
          }
          break;
        case SDL_MOUSEBUTTONUP:
          drawing = 0;
          break;
        case SDL_MOUSEMOTION:
          SDL_Rect rect = {40, 160, 279, 279};
          if (insideRegion(rect, e.motion.x, e.motion.y)) {
            painelMouseMotionHandler(renderer, e.motion.x, e.motion.y, drawing);
          }
      }

    }
    SDL_RenderPresent(renderer);
  }
}

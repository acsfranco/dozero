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
  
  drawPainel(renderer);
  printtext(renderer, 400, 20, "-- Do Zero --", white, 1, 0, 36);
  printtext(renderer, 400, 80, "Reconhecimento de digitos manuscritos", white, 1, 0, 36);
  printtext(renderer, 180, 130, "Desenhe seu digito aqui", white, 1, 0, 22);

  uint8_t running = 1;
  while (running) {
    while (SDL_PollEvent(&e)) {
      switch(e.type) {
        case SDL_QUIT:
          running = 0;
          break;
      }

    }
    SDL_RenderPresent(renderer);
  }
}

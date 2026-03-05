#include<SDL2/SDL.h>
#include<SDL2/SDL_ttf.h>
#include<stdio.h>
#include<stdint.h>

#include "../../redesneurais/src/versao2-backprop/neuron.h"
#include "../../redesneurais/src/versao2-backprop/neuralnet.h"
#include "../../redesneurais/src/versao2-backprop/netmath.h"

// CORES PREDEFINIDAS
SDL_Color white = {255, 255, 255, 255};
SDL_Color black = {0, 0, 0, 255};
SDL_Color yellow = {255, 255, 0, 255};
SDL_Color cyan = {0, 255, 255, 255};
SDL_Color blue = {0, 0, 255, 255};
SDL_Color green = {0, 255, 0, 255};

// ENTRADA DA REDE NEURAL
float input[784];

/*  FUNÇÃO: drawPainel:
 *
 *
 *
 *
 *
 *

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

void resetInput(SDL_Renderer *renderer) {
  for (uint16_t i = 0; i < 784; i++)
    input[i] = 0;
  float percents[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  showresult(renderer, percents);
}

void cleanButtonClickHandle(SDL_Event e, SDL_Renderer *renderer) {
  drawPainel(renderer);
  resetInput(renderer);
}

void recognizeButtonClickHandle(SDL_Event e) {
  printf("Em construção!!!\n");
}

void drawpixel(SDL_Renderer *renderer, uint16_t mx, uint16_t my, uint8_t color) {
  SDL_Rect rect = {(mx / 10) * 10, (my / 10) * 10, 10, 10};
  SDL_SetRenderDrawColor(renderer, 0, color, color, 255);
  SDL_RenderFillRect(renderer, &rect);
}

void blurinput(uint32_t x, uint32_t y) {
  float c = input[y * 28 + x] * 1.5f;
  if (x + 1 < 28) {
    c += input[y * 28 + x + 1];
  }

  if (x - 1 >= 0) {
    c += input[y * 28 + x - 1];
  }
  
  if (y + 1 < 28) {
    c += input[(y + 1) * 28 + x];
  }
  
  if (y - 1 >= 0) {
    c += input[(y - 1) * 28 + x];
  }

  c /= 4.0f;

  if (c > 1.0f)
    c = 1.0f;

  input[y * 28 + x] = c;
}

void setInput(SDL_Renderer *renderer, uint16_t x, uint16_t y, uint16_t mx, uint16_t my) {
  input[y * 28 + x] = 1.0f;
  blurinput(x, y);
  drawpixel(renderer, mx, my, input[y * 28 + x] * 255.0f);
  if (x + 1 < 28) {
    blurinput(x + 1, y);
    drawpixel(renderer, mx + 10, my, input[y * 28 + x + 1] * 255.0f);
  }
  
  if (x - 1 >= 0) {
    blurinput(x - 1, y);
    drawpixel(renderer, mx - 10, my, input[y * 28 + x - 1] * 255.0f);
  }

  if (y + 1 < 28) {
    blurinput(x, y + 1);
    drawpixel(renderer, mx, my + 10, input[(y + 1) * 28 + x] * 255.0f);
  }

  if (y - 1 >= 0) {
    blurinput(x, y - 1);
    drawpixel(renderer, mx, my - 10, input[(y - 1) * 28 + x] * 255.0f);
  } 
}

void painelMouseMotionHandler(SDL_Renderer *renderer, uint16_t mx, uint16_t my, uint8_t drawing, NET net) {
  if (drawing) { 
    uint16_t x = (mx - 40.0f) / 10.0f, y = (my - 160.0f) / 10;
    setInput(renderer, x, y, mx, my);
    float *out = feedforward(net, input);
    showresult(renderer, out);
  }
}

void initializeIA(NET *net) {
  ACTFUNC actsig = {sig, derivsig, 0};
  uint32_t layers[] = {784, 64, 10};
  *net = initnet(layers, 3, actsig, actsig);
  loadweights(net->outneurons, net->nout, NULL, "../../redesneurais/src/versao2-backprop/neuralweights.net");
}


void main() {
  NET net;
  initializeIA(&net);

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
  
  resetInput(renderer);
  drawPainel(renderer);
  printtext(renderer, 400, 20, "-- Do Zero --", white, 1, 0, 36);
  printtext(renderer, 400, 80, "Reconhecimento de digitos manuscritos", white, 1, 0, 36);
  printtext(renderer, 180, 130, "Desenhe seu digito aqui", white, 1, 0, 22);
  SDL_Rect rect_c = {39, 460, 282, 60};
  SDL_Rect rect_r = {39, 530, 282, 60};
  createbutton(renderer, rect_c, green, black, "Limpar");
  createbutton(renderer, rect_r, blue, white, "Reconhecer");
 
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
            painelMouseMotionHandler(renderer, e.motion.x, e.motion.y, drawing, net);
          }
      }

    }
    SDL_RenderPresent(renderer);
  }
}

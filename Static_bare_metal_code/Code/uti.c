#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "uti.h"

void data_shuffle(Data *d){
    for(int i = d->rows - 1; i>0; i--){
        int j = rand() % (i+1);

        for(int k=0; k<d->nops; k++){
            float temp = d->tg[i][k];
            d->tg[i][k] = d->tg[j][k];
            d->tg[j][k] = temp;
        }

        for(int m=0; m<MAX_ROWS; m++){
            for(int n=0; n<MAX_ROWS; n++){
                float temp = d->in[i][m][n];
                d->in[i][m][n] = d->in[j][m][n];
                d->in[j][m][n] = temp;
            }
        }
    }
}

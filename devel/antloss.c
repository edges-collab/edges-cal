/*
    File for computing antenna losses for EDGES-2.

    The functions here are simply copied from edges2k.c, and a new main() is written
    so that losses can be easily computed for testing.

    We only keep mode = 5,6,7 for lossmodel (as that is for EDGES-2 lowband, and we
    want to simplify the code for cross-comparison).
*/

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/io.h>
#include <time.h>
#define PI 3.1415926536
#define TWOPI 6.28318530717958
//#define VERBOSE

double lossmodel(double, double, double, int);
complex double cabl2(double, double, complex double, int, complex double *, complex double *, double);

double lossmodel(double freq, double s11arl, double s11aim, int mode) {
  complex double s11, s22, s12, s21, ta11, ta12, ta21, ta22, tb11, tb12, tb21, tb22;
  complex double t11, t12, t21, t22, T, s11a;
  double del, del4, L2, ttest, ttest2;

  int m;
  ttest = ttest2 = 1;                                                                              // ttest = 0.5;
  del4 = 1e-10;                                                                                    // lowband Fairview SC3792
  if (mode == 5 || mode == 6 || mode == 7 || mode == 8 || mode == 9) del = 43.6 * 2.54e-2 / 3e08;  // balun lowband 43"

  if (mode == 5 || mode == 6 || mode == 7 || mode == 8 || mode == 9 || mode == 10 || mode == 11) {
    cabl2(freq, del4, 0, 8, &s11, &s12, ttest);
    s21 = s12;
    s22 = s11;
  }  // SC3792
  ta11 = -(s11 * s22 - s12 * s21) / s21;
  ta12 = s11 / s21;
  ta21 = -s22 / s21;
  ta22 = 1 / s21;  // close to ref. plane
#ifdef VERBOSE
    printf(
        "ta11: %1.12e $%1.12e\nta12: %1.12e %1.12e\nta21: %1.12e %1.12e\nta22 %1.12e %1.12e\n",
        creal(ta11), cimag(ta11), creal(ta12), cimag(ta12), 
        creal(ta21), cimag(ta21), creal(ta22), cimag(ta22)
    );
#endif

  if (mode == 5 || mode == 6 || mode == 7 || mode == 8 || mode == 9) {
    cabl2(freq, del, 0, 5, &s11, &s12, ttest2);
    s21 = s12;
    s22 = s11;
  }  // low band balun tube

  tb11 = -(s11 * s22 - s12 * s21) / s21;
  tb12 = s11 / s21;
  tb21 = -s22 / s21;
  tb22 = 1 / s21;  // T matrix from s-parms closer to antenna

#ifdef VERBOSE  
  printf(
    "tb11: %1.12e %1.12e\ntb12: %1.12e %1.12e\ntb21: %1.12e %1.12e\ntb22 %1.12e %1.12e\n",
    creal(tb11), cimag(tb11), creal(tb12), cimag(tb12), 
    creal(tb21), cimag(tb21), creal(tb22), cimag(tb22) 
);
#endif

  t11 = ta11 * tb11 + ta12 * tb21;
  t12 = ta11 * tb12 + ta12 * tb22;
  t21 = ta21 * tb11 + ta22 * tb21;
  t22 = ta21 * tb12 + ta22 * tb22;  // A*B

#ifdef VERBOSE
printf(
    "t11: %1.12e %1.12e\nt12: %1.12e %1.12e\nt21: %1.12e %1.12e\nt22 %1.12e %1.12e\n",
    creal(t11), cimag(t11), creal(t12), cimag(t12), 
    creal(t21), cimag(t21), creal(t22), cimag(t22) 
);
#endif

  s11 = t12 / t22;
  s12 = (t11 * t22 - t12 * t21) / t22;
  s21 = 1 / t22;
  s22 = -t21 / t22;

  s11a = s11arl + I*s11aim;
  T = (s11a - s11) / (s12 * s21 - s11 * s22 + s22 * s11a);  // from memo 132
  L2 = cabs(s12 * s21) * (1 - T * conj(T)) / ((1 - s11a * conj(s11a)) * (1 - s22 * T) * (1 - conj(s22 * T)));
  return L2;
}

complex double cabl2(
    double freq, double delay, complex double Tin, int mode, complex double *ss11, 
    complex double *ss12, double ttest
) {
  complex double T, Zcab, g, s11, s12, s21, s22, Vin, Iin, Vout, VVin, Z;
  double a, b, d, d2, diel, R, C, L, La, Lb, disp, G;
  // ttest = 0.5;  // best fit from day 2015_023
  if (mode == 5) {
    b = 0.75 * 2.54e-2 * 0.5;
    a = (5.0 / 16.0) * 2.54e-2 * 0.5;
    diel = 1.07;                                               // balun tube lowband
    d2 = sqrt(1.0 / (PI * 4.0 * PI * 1e-7 * 5.96e07 * 0.29));  // skin depth at 1 Hz for brass
    d = sqrt(1.0 / (PI * 4.0 * PI * 1e-7 * 5.96e07 * ttest));  // skin depth at 1 Hz for copper
  }
  if (mode == 8) {
    b = 0.161 * 2.54e-2 * 0.5;
    a = 0.05 * 2.54e-2 * 0.5;
    diel = 2.05;                                                        // SC3792 connector - new dimensions from Fairviwe 8 Dec 15
    d2 = sqrt(1.0 / (PI * 4.0 * PI * 1e-7 * 5.96e07 * 0.024 * ttest));  // for Stainless
    d = sqrt(1.0 / (PI * 4.0 * PI * 1e-7 * 5.96e07 * 0.24 * ttest));    // skin depth at 1 Hz for brass - might be less 0.24
  }

  L = (4.0 * PI * 1e-7 / (2.0 * PI)) * log(b / a);
  C = 2.0 * PI * 8.854e-12 * diel / log(b / a);

  La = 4.0 * PI * 1e-7 * d / (4.0 * PI * a);
  Lb = 4.0 * PI * 1e-7 * d2 / (4.0 * PI * b);
  disp = (La + Lb) / L;
  R = 2.0 * PI * L * disp * sqrt(freq * 1e6);
  L = L * (1.0 + disp / sqrt(freq * 1e6));
  G = 0;
  if (diel > 1.2) G = 2.0 * PI * C * freq * 1e6 * 2e-4;
  Zcab = csqrt((I * 2 * PI * freq * 1e6 * L + R) / (I * 2 * PI * freq * 1e6 * C + G));
  g = csqrt((I * 2 * PI * freq * 1e6 * L + R) * (I * 2 * PI * freq * 1e6 * C + G));
#ifdef VERBOSE
  printf("SKIN DEPTH: %lf %lf\n", d2, d);
  printf("PROP CONSTANT: %lf %lf\n", creal(g), cimag(g));
  printf("G: %1.12e\n", G);
  printf("C: %1.12e\n", C);
  printf("L: %1.12e\n", L);
  printf("R: %1.12e\n", R);
#endif
  T = (50.0 - Zcab) / (50.0 + Zcab);
  Vin = (cexp(+g * delay * 3e08) + T * cexp(-g * delay * 3e08));
  Iin = (cexp(+g * delay * 3e08) - T * cexp(-g * delay * 3e08)) / Zcab;
  Vout = (1 + T);  // Iout = (1 - T)/Zcab;
  s11 = s22 = ((Vin / Iin) - 50) / ((Vin / Iin) + 50);
  VVin = Vin + 50.0 * Iin;
  s12 = s21 = (2 * Vout / VVin);
  *ss11 = s11;
  *ss12 = s12;
#ifdef VERBOSE
  printf("s11: %1.12e %1.12e\n", creal(s11), cimag(s11));
  printf("s12: %1.12e %1.12e\n", creal(s12), cimag(s12));
#endif
  Z = 50.0 * (1 + Tin) / (1 - Tin);
  T = (Z - Zcab) / (Z + Zcab);
  T = T * cexp(-g * 2 * delay * 3e08);
  Z = Zcab * (1 + T) / (1 - T);
  T = (Z - 50.0) / (Z + 50.0);
  return T;
}

void main(int argc, char *argv[]){
    double freq;   // In MHz
    double s11rl, s11im;

    sscanf(argv[1], "%lf", &freq);
    sscanf(argv[2], "%lf", &s11rl);
    sscanf(argv[3], "%lf", &s11im);


    double loss = lossmodel(freq, s11rl, s11im, 6);
    #ifdef VERBOSE
    printf("Freq = %1.12e\n", freq);
    printf("s11rl = %1.12e\n", s11rl);
    printf("s11im = %1.12e\n", s11im);
    #endif
    
    printf("Got loss %1.12e\n", loss);
}
#ifndef PARTICLE_FILTER
#define PARTICLE_FILTER

#include "colorFeatures.h"
#include <time.h>

#define TRANS_X_STD 0.5
#define TRANS_Y_STD 1.0
#define TRANS_S_STD 0.001
#define MAX_PARTICLES 50

#define SHOW_ALL 0
#define SHOW_SELECTED 1

#define A1 2.0
#define A2 -1.0
#define B0 1.0000

typedef struct particle {
    float x; /** current x coordinate */
    float y; /** current y coordinate */
    float s; /** scale */
    float xp; /** previous x coordinate */
    float yp; /** previous y coordinate */
    float sp; /** previous scale */
    float x0; /** original x coordinate */
    float y0; /** original y coordinate */
    int width; /** original width of region described by particle */
    int height; /** original height of region described by particle */
    float w; /** weight*/
} particle;

/** Ali and M. N. Dailey, "Multiple human tracking in high-density crowds"
 Abschnitt 2.3 Particle filter */
class particleFilter {
public:
    particleFilter();
    ~particleFilter();
    void resetCCWeights();
    CvRect getParticleRect();
    CvPoint getParticleCenter();
    particle particles[MAX_PARTICLES];
    /** Initializes particles */
    void initParticles(CvRect region, int particlesPerObject);
    /** Reset particles */
    void resetParticles(CvRect region);
    /** Moves particles */
    void transition(int w, int h);
    /** Normalize weights of particles */
    void normalizeWeights();
    /** Resamples particles */
    void resample();
    void displayParticles(IplImage* img, CvScalar nColor, CvScalar hColor, int param);
    /** Updates particle weights */
    void updateWeight(IplImage* frameHSV, histogram* objectHisto);
    
    int nParticles;
    gsl_rng* rng;
    float weight;
    int objectID;
private:
    particle calTransition(particle p, int w, int h, gsl_rng* rng);
    void displayParticle(IplImage* img, particle p, CvScalar color);
};

#endif
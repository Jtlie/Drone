/**
 * @copyright
 *
 * Copyright 2012 Kevin Schluff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2, 
 * as published by the Free Software Foundation.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */

#include "filter.h"
#include "hist.h"
#include <iostream>
using namespace cv;
using namespace std;
typedef unsigned int uint;

ParticleFilter::ParticleFilter(int num_particles)
   :ConDensation(NUM_STATES, num_particles),
    m_mean_confidence(0.f)
{}

ParticleFilter::~ParticleFilter()
{}

void ParticleFilter::init(const Rect& selection)
{
   static const float DT = 1;

   // Constant velocity model with constant scale
   m_transition_matrix = (Mat_<float>(NUM_STATES, NUM_STATES) << 
				  1, 0, DT,  0,  0,
				  0, 1,  0, DT,  0,
				  0, 0,  1,  0,  0,
				  0, 0,  0,  1,  0,
				  0, 0,  0,  0,  1);

   const float initial[NUM_STATES] = {selection.x + selection.width/2, selection.y + selection.height/2, 0, 0, 1.0};
   static const float std_dev[NUM_STATES] = { 2,  2,  .5,  .5,  .1};
 
   cout << "Init with state: [ ";
   for( uint j = 0; j < NUM_STATES; j++)
   {
      cout << initial[j] << " ";
   }
   cout << "]" << endl;

   init_sample_set(initial, std_dev);		
}

/**
 * Update filter with measurements and time step. 
 */
Mat& ParticleFilter::update(Mat& image, Mat& lbp_image, const Size& target_size, Mat& target_hist, bool use_lbp)
{
   Mat hist;
   Rect bounds(0,0,image.cols, image.rows);

   // Update the confidence for each particle
   for( uint i = 0; i< m_num_particles; i++)
   {
      float scale = MAX(0.1, m_particles[i](STATE_SCALE));
      m_particles[i](STATE_SCALE) = scale;
      int width = round(target_size.width * scale);
      int height = round(target_size.height * scale);
      int x = round(m_particles[i](STATE_X)) - width / 2;
      int y = round(m_particles[i](STATE_Y)) - height / 2;

      Rect region = Rect(x, y, width, height) & bounds;
      Mat image_roi(image, region), lbp_roi(lbp_image, region);

      m_confidence[i] = calc_likelyhood(image_roi, lbp_roi, target_hist, use_lbp);
   }

   // Project the state forward in time
   time_update();

   // Update the confidence at the mean state
   float scale = MAX(0.1, m_state(STATE_SCALE));
   m_state(STATE_SCALE) = scale;
   int width = round(target_size.width * scale);
   int height = round(target_size.height * scale);
   int x = round(m_state(STATE_X)) - width / 2;
   int y = round(m_state(STATE_Y)) - height / 2;

   Rect region = Rect(x, y, width, height) & bounds;
   Mat image_roi(image, region), lbp_roi(lbp_image, region);

   m_mean_confidence = calc_likelyhood(image_roi, lbp_roi, target_hist, use_lbp);

   // Redistribute particles to reacquire the target if the mean state moves 
   // off screen.  This usually means the target has been lost due to a mismatch
   // between the modelled motion and actual motion.
   if( !bounds.contains(Point(round(m_state(STATE_X)), round(m_state(STATE_Y)))) )
   {
      static const float lower_bound[NUM_STATES] = {0, 0, -.5, -.5, 1.0};
      static const float upper_bound[NUM_STATES] = {image.cols, image.rows, .5, .5, 2.0};
   
      cout << "Redistribute: " << m_state << " " << m_mean_confidence << endl;
      redistribute( lower_bound, upper_bound );
   }
 
   return m_state;
}


// Calculate the likelyhood for a particular region
float ParticleFilter::calc_likelyhood(Mat& image_roi, Mat& lbp_roi, Mat& target_hist, bool use_lbp )
{
   static const float LAMBDA = 20.f;
   static Mat hist;

   calc_hist(image_roi, lbp_roi, hist, use_lbp);
   normalize(hist, hist);

   float bc = compareHist(target_hist, hist, CV_COMP_BHATTACHARYYA);
   float prob = 0.f;
   if(bc != 1.f) // Clamp total mismatch to 0 likelyhood
      prob = exp(-LAMBDA * (bc * bc) );
   return prob;
}

void ParticleFilter::draw_estimated_state(Mat& image, const Size& target_size, const Scalar& color)
{
   Rect bounds(0,0, image.cols, image.rows);

   for(uint i = 0; i < m_num_particles; i++)
   {
      int width = round(target_size.width * m_state(STATE_SCALE));
      int height = round(target_size.height * m_state(STATE_SCALE));
      int x = round(m_state(STATE_X)) - width/2;
      int y = round(m_state(STATE_Y)) - height/2;
      Rect rect = Rect(x, y, width, height) & bounds;
      rectangle(image, rect, color, 2);
   }
}

void ParticleFilter::draw_particles(Mat& image, const Size& target_size, const Scalar& color)
{
   Rect bounds(0,0, image.cols, image.rows);

   for(uint i = 0; i < m_num_particles; i++)
   {
      int width = round(target_size.width * m_particles[i](STATE_SCALE));
      int height = round(target_size.height * m_particles[i](STATE_SCALE));
      int x = round(m_particles[i](STATE_X)) - width/2;
      int y = round(m_particles[i](STATE_Y)) - height/2;
      Rect rect = Rect(x, y, width, height) & bounds;
      rectangle(image, rect, color, 1);
   }
}

void ParticleFilter::redistribute(const float lbound[], const float ubound[] )
{
   for( uint i = 0; i < m_num_particles; i++ )
   {
      for( uint j = 0; j < m_num_states; j++ )
      {
	 float r = m_rng.uniform(lbound[j], ubound[j]);
	 m_particles[i](j) = r;
      }

      m_confidence[i] = 1.0 / (float)m_num_particles;
   }

}

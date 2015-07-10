#ifndef CONDENS_H
#define CONDENS_H
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

#include <opencv2/opencv.hpp>

class ConDensation
{
public:

   ConDensation( unsigned int dynam_params,
		 unsigned int num_particles );

   virtual ~ConDensation();

   void init_sample_set(const float initial[], const float std_dev[] );

   void time_update();
   

protected:

   unsigned int m_num_states;
   cv::Mat_<float> m_transition_matrix;   // Matrix of the linear system  
   cv::Mat_<float>  m_state;              // Vector of current State
   unsigned int m_num_particles;          //  Number of the Samples 
   std::vector<cv::Mat_<float> > m_particles;      // Current particle vectors
   std::vector<float> m_confidence;      // Confidence for each particle vector 
   std::vector<cv::Mat_<float> > m_new_particles;  // New samples
   std::vector<float> m_cumulative;      // Cumulative confidence vector    
   cv::Mat_<float> m_temp;               // Temporary vector 
   cv::RNG m_rng;                        // Random generator
   const float* m_std_dev;
   
};
                               

#endif

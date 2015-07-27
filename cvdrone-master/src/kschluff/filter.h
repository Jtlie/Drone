#ifndef PARTICLEFILTER_H
#define PARTICLEFILTER_H
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
#include "condens.h"

class ParticleFilter : private ConDensation
{
public:

   enum FilterStates
   {
      STATE_X,
      STATE_Y,
      STATE_X_VEL,
      STATE_Y_VEL,
      STATE_SCALE,
      NUM_STATES
   };

   ParticleFilter(int num_particles);
   virtual ~ParticleFilter();

   void init(const cv::Rect& selection);

   cv::Mat& update(cv::Mat& image, cv::Mat& lbp_image, const cv::Size& target_size, cv::Mat& target_hist, bool use_lbp);

   void draw_estimated_state(cv::Mat& image, const cv::Size& target_size, const cv::Scalar& color);

   void draw_particles(cv::Mat& image, const cv::Size& target_size, const cv::Scalar& color);

   void redistribute(const float lower_bound[], const float upper_bound[]);

   const cv::Mat& state() const
   { return m_state; }

   float confidence() const
   { return m_mean_confidence; };

private:
   
   float calc_likelyhood(cv::Mat& image_roi, cv::Mat& lbp_roi, cv::Mat& target_hist, bool use_lbp );

   float m_mean_confidence;
   


};

#endif

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
#ifndef LBP_H
#define LBP_H

#include <opencv2/opencv.hpp>

void lbp_init(bool uniform = true);

unsigned int lbp_num_patterns();

void lbp_from_gray(const cv::Mat& image, cv::Mat& lbp_image);

cv::Mat lbp_histogram(const cv::Mat& lbp_image, const cv::Rect& selection, bool norm = true);


#endif

/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 5;  // TODO: Set the number of particles
  auto x_dist = std::normal_distribution<double>(x, std[0]);
  auto y_dist = std::normal_distribution<double>(y, std[1]);
  auto theta_dist = std::normal_distribution<double>(theta, std[2]);
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x_dist(gen);
    p.y = y_dist(gen);
    p.theta = theta_dist(gen);
    p.weight = 1;
    particles.push_back(p);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  auto x_dist = std::normal_distribution<double>(0, std_pos[0]);
  auto y_dist = std::normal_distribution<double>(0, std_pos[1]);
  auto theta_dist = std::normal_distribution<double>(0, std_pos[2]);
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    Particle &p = particles[i];

    p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + x_dist(gen);
    p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + y_dist(gen);
    p.theta += yaw_rate * delta_t + theta_dist(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (Particle &p: particles) {
    double weight = 1;
    double sin_p_theta = sin(p.theta);
    double cos_p_theta = cos(p.theta);

    /*
    std::vector<Map::single_landmark_s> landmarks_in_range;
    for (const auto &landmark: map_landmarks.landmark_list) {
      double dx = landmark.x_f - p.x;
      double dy = landmark.y_f - p.y;
      double dist = sqrt(dx * dx + dy * dy);

      if (dist < sensor_range)
        landmarks_in_range.push_back(landmark);
    }*/

    for (auto &o: observations) {
      // Transform observation into map coordinates
      double map_x = p.x + cos_p_theta * o.x - sin_p_theta * o.y;
      double map_y = p.y + sin_p_theta * o.x + cos_p_theta * o.y;

      double min_dist = sensor_range;
      int landmark_idx = -1;

      // Find nearest landmark
      for (const auto &landmark: map_landmarks.landmark_list) {
        double dx = landmark.x_f - map_x;
        double dy = landmark.y_f - map_y;
        double dist = (dx * dx + dy * dy); // Squared distance, saves a sqrt

        if (dist < min_dist) {
          landmark_idx = landmark.id_i - 1;
          min_dist = dist;
        }
      }

      weight *= multiv_prob(map_x, map_y,
                            map_landmarks.landmark_list[landmark_idx].x_f,
                            map_landmarks.landmark_list[landmark_idx].y_f,
                            std_landmark[0], std_landmark[1]);
    }

    p.weight = weight;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::vector<double> weights;

  for (auto &p: particles) {
    weights.push_back(p.weight);
  }

  std::vector<Particle> newParticles(particles.size());
  std::discrete_distribution<> d(weights.begin(), weights.end());
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    newParticles.push_back(particles[d(gen)]);
  }

  particles = std::move(newParticles); // std::move faster than copy
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
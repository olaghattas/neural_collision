

/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <chrono>
#include <thread>
#include <unistd.h>

#include "particle_filter.h"

#define NUM_PARTICLES 200

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // initialize distribution generators
    std::default_random_engine generator;
    std::normal_distribution<double> x_dist(x, std[0]);
    std::normal_distribution<double> y_dist(y, std[1]);
    std::normal_distribution<double> theta_dist(theta, std[2]);

    // set the number of particles to be generated
    num_particles = NUM_PARTICLES;

    // create and initialize each particle and add to the vector
    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = x_dist(generator);
        p.y = y_dist(generator);
        p.theta = theta_dist(generator);
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(1.0);
    }

    // set the initialization flag to true
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    // calculate frequently used values
    double motion_factor_1;
    if (fabs(yaw_rate) > 0.001) {
        motion_factor_1 = velocity / yaw_rate;
    } else {
        motion_factor_1 = velocity;
    }
    double motion_factor_2 = yaw_rate * delta_t;

    // initialize distribution generators
    std::default_random_engine generator;
    std::normal_distribution<double> x_dist(0, std_pos[0]);
    std::normal_distribution<double> y_dist(0, std_pos[1]);
    std::normal_distribution<double> theta_dist(0, std_pos[2]);

    // for each of the particle predict the new position by applying the motion model
    for (int i = 0; i < particles.size(); ++i) {

        double delta_theta = particles[i].theta + motion_factor_2;
        particles[i].x = particles[i].x + (motion_factor_1 * (sin(delta_theta) - sin(particles[i].theta)));
        particles[i].y = particles[i].y + (motion_factor_1 * (cos(particles[i].theta) - cos(delta_theta)));

        particles[i].x = particles[i].x + x_dist(generator);
        particles[i].y = particles[i].y + y_dist(generator);
        particles[i].theta = delta_theta + theta_dist(generator);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    // for each of the observation fetch the closest predicted particle
    for (int i = 0; i < observations.size(); ++i) {

        double min_distance = 9999999;
        int landmark_id = -1;

        // loop through each of the particle to find the nearest
        for (int j = 0; j < predicted.size(); j++) {
            double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
            // if distance is minimus so far save the landmark id
            if (distance < min_distance) {
                landmark_id = predicted[j].id;
                min_distance = distance;
            }
        }

        // add only if a valid landmark was found
        if (landmark_id != -1) {
            observations[i].id = landmark_id;
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html

    // clear the previous set weights
    weights.clear();

    // loop through each of the particle to update
    for (int i_p = 0; i_p < particles.size(); i_p++) {

        // transform the observations to MAP's coordinate system
        std::vector<LandmarkObs> transformed_observations;
        for (int i_o = 0; i_o < observations.size(); ++i_o) {
            LandmarkObs observation = observations[i_o];
            LandmarkObs transformed_observation;

            transformed_observation.id = observation.id;
            transformed_observation.x =
                    (observation.x * cos(particles[i_p].theta)) - (observation.y * sin(particles[i_p].theta)) +
                    particles[i_p].x;
            transformed_observation.y =
                    (observation.x * sin(particles[i_p].theta)) + (observation.y * cos(particles[i_p].theta)) +
                    particles[i_p].y;

            transformed_observations.push_back(transformed_observation);
        }

        // only select the landmarks that are at a valid distance
        std::vector<LandmarkObs> predicted;
        for (int i_pr = 0; i_pr < map_landmarks.landmark_list.size(); ++i_pr) {
            Map::single_landmark_s item = map_landmarks.landmark_list[i_pr];
            double distance = dist(particles[i_p].x, particles[i_p].y, item.x_f, item.y_f);
            if (distance <= sensor_range) {
                LandmarkObs landmark;
                landmark.id = item.id_i;
                landmark.x = item.x_f;
                landmark.y = item.y_f;

                predicted.push_back(landmark);
            }
        }

        // associate landmarks to each of the transformed observation
        dataAssociation(predicted, transformed_observations);

        // initialize values
        double weight = 1.0;
        double sigma_x = std_landmark[0];
        double sigma_y = std_landmark[1];
        double gaussian_factor = 1 / (2 * M_PI * sigma_x * sigma_y);

        // loop through each of the landmark
        for (int j_o = 0; j_o < predicted.size(); j_o++) {

            // initialize values
            double mu_x = predicted[j_o].x;
            double mu_y = predicted[j_o].y;
            double min_distance = 999999;
            double x;
            double y;
            bool found = false;

            // loop through each of the observation to find the one closest to the landmark
            LandmarkObs neighbor;
            for (int k = 0; k < transformed_observations.size(); ++k) {
                if (transformed_observations[k].id == predicted[j_o].id) {
                    double distance = dist(transformed_observations[k].x, transformed_observations[k].y, mu_x, mu_y);
                    if (distance < min_distance) {
                        found = true;
                        min_distance = distance;
                        x = transformed_observations[k].x;
                        y = transformed_observations[k].y;
                    }
                }
            }

            // calculate the gaussian if a neighbor was found
            if (found) {

                double gaussian = ((x - mu_x) * (x - mu_x) / (2 * sigma_x * sigma_x)) +
                                  ((y - mu_y) * (y - mu_y) / (2 * sigma_y * sigma_y));
                gaussian = exp(-gaussian);
                gaussian = gaussian * gaussian_factor;

                weight = weight * gaussian;
            }
        }

        // add the weights to the vector
        weights.push_back(weight);
        particles[i_p].weight = weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // initialize distribution generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> distribution_weights(weights.begin(), weights.end());

    // select num_particles new particle based on the weights
    std::vector<Particle> resampled;
    for (int i = 0; i < num_particles; ++i) {
        int index = distribution_weights(gen);
        resampled.push_back(particles[index]);
    }

    // assign the new set of particles
    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
// api.h
#ifndef API_H
#define API_H

#include <curl/curl.h>
#include <cjson/cJSON.h>

// Declare the function for making the HTTP request
cJSON* observation();
int control(double vx, double vy, double angle, int state);

#endif // API_H
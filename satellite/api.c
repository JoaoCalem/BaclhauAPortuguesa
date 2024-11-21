#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

// Structure to hold the response data
struct MemoryStruct {
    char *memory;
    size_t size;
};

// Callback function for curl to write data into the MemoryStruct
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realSize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;

    char *ptr = realloc(mem->memory, mem->size + realSize + 1);
    if (ptr == NULL) {
        printf("Not enough memory (realloc returned NULL)\n");
        return 0;
    }

    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realSize);
    mem->size += realSize;
    mem->memory[mem->size] = 0;

    return realSize;
}

cJSON* observation() {
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;

    chunk.memory = malloc(1);
    chunk.size = 0;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://10.100.10.2:33000/observation");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
            curl_easy_cleanup(curl);
            free(chunk.memory);
            curl_global_cleanup();
            return NULL;
        }

        cJSON *json = cJSON_Parse(chunk.memory);

        curl_easy_cleanup(curl);
        free(chunk.memory);
        curl_global_cleanup();

        return json;  // Return the parsed JSON object (or NULL on error)
    }

    curl_global_cleanup();
    return NULL;
}

int control(double vx, double vy, double angle, int state) {
    CURL *curl;
    CURLcode res;

    // Construct JSON payload
    cJSON *json = cJSON_CreateObject();
    if (!json) {
        fprintf(stderr, "Error creating JSON object\n");
        return -1;
    }
    cJSON_AddNumberToObject(json, "vel_x", vx);
    cJSON_AddNumberToObject(json, "vel_y", vy);
    cJSON_AddNumberToObject(json, "camera_angle", angle);
    cJSON_AddNumberToObject(json, "state", state);

    // Convert JSON object to string
    char *json_data = cJSON_PrintUnformatted(json);
    if (!json_data) {
        fprintf(stderr, "Error converting JSON to string\n");
        cJSON_Delete(json);
        return -1;
    }

    // Initialize libcurl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        // Set options for the PUT request
        curl_easy_setopt(curl, CURLOPT_URL, "http://10.100.10.2:33000/control");
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data);

        // Set HTTP headers
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Perform the request
        res = curl_easy_perform(curl);

        // Check for errors
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            printf("PUT request sent successfully\n");
        }

        // Clean up
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    // Free resources
    cJSON_Delete(json);
    free(json_data);
    curl_global_cleanup();

    return (res == CURLE_OK) ? 0 : -1;
}
#include <stdio.h>
#include <stdlib.h>
#include "api.h"

int main() {
    cJSON *json = observation();

    if (json) {
        // Example: Print the formatted JSON
        char *jsonStr = cJSON_Print(json);
        if (jsonStr) {
            printf("JSON Response:\n%s\n", jsonStr);
            free(jsonStr);
        }

        // // Example: Access a specific field (e.g., "name")
        // cJSON *name = cJSON_GetObjectItem(json, "name");
        // if (cJSON_IsString(name)) {
        //     printf("Name: %s\n", name->valuestring);
        // }

        cJSON_Delete(json);  // Free the JSON object
    } else {
        printf("Failed to fetch or parse JSON data\n");
    }

    return 0;
}
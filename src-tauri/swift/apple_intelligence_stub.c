#include <stdlib.h>
#include <string.h>

#include "apple_intelligence_bridge.h"

static char *copy_cstr(const char *src) {
  if (src == NULL) {
    return NULL;
  }

  size_t len = strlen(src) + 1;
  char *dst = (char *)malloc(len);
  if (dst == NULL) {
    return NULL;
  }
  memcpy(dst, src, len);
  return dst;
}

int is_apple_intelligence_available(void) { return 0; }

AppleLLMResponse *process_text_with_system_prompt_apple(const char *system_prompt,
                                                        const char *user_content,
                                                        int max_tokens) {
  (void)system_prompt;
  (void)user_content;
  (void)max_tokens;

  AppleLLMResponse *response = (AppleLLMResponse *)calloc(1, sizeof(AppleLLMResponse));
  if (response == NULL) {
    return NULL;
  }

  response->success = 0;
  response->response = NULL;
  response->error_message = copy_cstr(
      "Apple Intelligence is unavailable on this build. Install full Xcode to enable it.");

  if (response->error_message == NULL) {
    free(response);
    return NULL;
  }

  return response;
}

void free_apple_llm_response(AppleLLMResponse *response) {
  if (response == NULL) {
    return;
  }

  free(response->response);
  free(response->error_message);
  free(response);
}

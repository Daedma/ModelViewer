#pragma once

#include <iostream>
#include <vulkan/vulkan.hpp>

inline PFN_vkDebugUtilsMessengerCallbackEXT debugCallback = []( VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                                                                   VkDebugUtilsMessageTypeFlagsEXT              messageType,
                                                                   const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
                                                                   void *                                       pUserData ) -> VkBool32
{
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  return VK_FALSE;
};


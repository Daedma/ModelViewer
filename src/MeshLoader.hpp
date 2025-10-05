#pragma once
#include "VulkanMesh.hpp"

#include <string>

class ModelLoader
{
public:
    VulkanMesh loadOBJ(const std::string& filepath);
};

// src/test_VulkanContext.hpp
#include "VulkanContext.hpp"
#include "VulkanContextBuilder.hpp"

#include <GLFW/glfw3.h>
#include <gtest/gtest.h>

// Helper to check Vulkan/GLFW availability
bool isVulkanAvailable()
{
    if (!glfwInit())
        return false;
    uint32_t     count = 0;
    const char** exts  = glfwGetRequiredInstanceExtensions(&count);
    return count > 0 && exts != nullptr;
}

TEST(VulkanContextBuilderTest, BuildDefaultContext)
{
    VulkanContextBuilder builder;
    if (!isVulkanAvailable())
    {
        GTEST_SKIP() << "Vulkan/GLFW not available";
    }
    EXPECT_NO_THROW({
        VulkanContext ctx = builder.build();
        EXPECT_TRUE(*ctx.getInstance());
        EXPECT_TRUE(*ctx.getPhysicalDevice());
        EXPECT_TRUE(*ctx.getDevice());
        EXPECT_TRUE(*ctx.getGraphicsQueue());
    });
}

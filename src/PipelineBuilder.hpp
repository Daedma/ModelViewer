#pragma once
#include <string>
#include <vulkan/vulkan_raii.hpp>

class PipelineBuilder
{
public:
    PipelineBuilder&   setShaders(vk::raii::Device&  device,
                                  const std::string& vertPath,
                                  const std::string& fragPath);
    PipelineBuilder&   setRenderPass(vk::RenderPass renderPass);
    PipelineBuilder&   setVertexInput(/* layout params */);
    vk::raii::Pipeline build(vk::raii::Device& device);

private:
    vk::PipelineShaderStageCreateInfo vertStage;
    vk::PipelineShaderStageCreateInfo fragStage;
    vk::RenderPass                    renderPass;
    // + другие параметры пайплайна
};

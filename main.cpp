#include "Renderer.hpp"
#include "SwapchainManager.hpp"
#include "VulkanContext.hpp"

int main()
{
  // Инициализация окна (GLFW/SDL и т.д.)
  GLFWwindow * window = initWindow();

  // Создание поверхности
  VkSurfaceKHR surface = createWindowSurface( instance, window );

  VulkanContext    context;
  SwapchainManager swapchain( context, surface, 1280, 720 );
  Renderer         renderer( context, swapchain );

  while ( !glfwWindowShouldClose( window ) )
  {
    glfwPollEvents();
    renderer.drawFrame();
  }

  context.getDevice().waitIdle();  // vk::raii: RAII всё почистит сам
}

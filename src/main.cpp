#define NOMINMAX // Grumble grumble Windows.h grumble grumble grumble

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <jitify.hpp>

#include <SFML/Graphics.hpp>

// Taken from jitify_example.hpp
#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    if (call != CUDA_SUCCESS) {                                           \
      const char* str;                                                    \
      cuGetErrorName(call, &str);                                         \
      std::cout << "(CUDA) returned " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      throw std::runtime_error("Ooops");                                  \
    }                                                                     \
  } while (0)

const auto floatsPerPixel = std::size_t{4}; // Technically only 3 are needed, but 4 aligns nicely (performance difference is untested)

void draw_image(std::vector<float>& out, std::size_t imgWidth, std::size_t imgHeight, int iterations, float angle, float power) {
    static jitify::JitCache kernel_cache;

    jitify::Program program = kernel_cache.program(
        "data/render.cuh",
        0,
        {"-std=c++11", "--use_fast_math", "-I" CUDA_INC_DIR}
    );

    out.resize(imgWidth * imgHeight * floatsPerPixel);

    float* d_data = nullptr;

    const auto total_bytes = out.size() * sizeof(float);

    // TODO: use cudaMallocPitch for more optimized 2D memory alignment
    cudaMalloc(reinterpret_cast<void**>(&d_data), total_bytes);
    cudaMemcpy(d_data, out.data(), total_bytes, cudaMemcpyHostToDevice);

    const auto grid_size = std::size_t{32};

    // TODO: invoke kernel in batches to support larger images

    dim3 grid(
        std::min(grid_size, imgWidth),
        std::min(grid_size, imgHeight)
    );
    dim3 block(
        1 + imgWidth / grid_size,
        1 + imgHeight / grid_size
    );

    CHECK_CUDA(program.kernel("renderKernel")
        .instantiate<int>(iterations)
        .configure(grid, block)
        .launch(d_data, imgWidth, imgHeight, angle, power)
    );

    cudaMemcpy(out.data(), d_data, total_bytes, cudaMemcpyDeviceToHost);

    // TODO: reuse the same buffer
    cudaFree(d_data);
}

int main() {
    auto angle = 0.0f;
    auto power = 8.0f;
    auto iterations = 512;

    const auto moveStep = 0.05f;
    const auto scaleRatio = 1.05f;

    const auto w = std::size_t{512};
    const auto h = std::size_t{512};
    auto img = sf::Image();
    img.create(static_cast<unsigned int>(w), static_cast<unsigned int>(h));
    std::vector<float> data;

    auto tex = sf::Texture();

    const auto update = [&](){
        draw_image(data, w, h, iterations, angle, power);
        assert(data.size() == (w * h * floatsPerPixel));
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                const auto r = std::clamp(data[floatsPerPixel * (x + y * w) + 0], 0.0f, 1.0f);
                const auto g = std::clamp(data[floatsPerPixel * (x + y * w) + 1], 0.0f, 1.0f);
                const auto b = std::clamp(data[floatsPerPixel * (x + y * w) + 2], 0.0f, 1.0f);
                auto ri = static_cast<std::uint8_t>(std::pow(r, 0.45f) * 255.0f);
                auto gi = static_cast<std::uint8_t>(std::pow(g, 0.45f) * 255.0f);
                auto bi = static_cast<std::uint8_t>(std::pow(b, 0.45f) * 255.0f);
                img.setPixel(x, y, sf::Color{ri, gi, bi});
            }
        }
        tex.loadFromImage(img);
    };

    auto win = sf::RenderWindow(
        sf::VideoMode{
            static_cast<unsigned int>(w),
            static_cast<unsigned int>(h)
        },
        "Path Tracer Demo"
    );

    update();
    auto sprite = sf::Sprite(tex);

    while (win.isOpen()) {
        auto e = sf::Event{};
        while (win.pollEvent(e)) {
            if (e.type == sf::Event::Closed) {
                win.close();
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
            angle -= moveStep;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
            angle += moveStep;
        }
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            iterations -= iterations % 16;
            iterations += 16;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
            iterations = std::max(1, iterations - 16);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
            power += 0.1f;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
            power -= 0.1f;
        }
        update();

        win.clear(sf::Color::Black);
        win.draw(sprite);
        win.display();
    }

    return 0;
}

#pragma once

// Minimal programmatic Cocoa window hosting a CAMetalLayer — no Xcode
// project, no storyboard. The C++ side casts the returned pointer to
// CA::MetalLayer* (metal-cpp) and drives the render loop itself.

extern "C" {

// Creates the window (size in points) and returns its CAMetalLayer*,
// configured for the given MTLDevice. Returns nullptr if no window
// server is available (headless session): callers must degrade
// gracefully.
void* lvCreateWindow(int width, int height, const char* title,
                     void* mtlDevice);

// Pumps pending UI events. Keys: space = pause, q = quit (or close the
// window). Returns 0 = quit requested, 1 = running, 2 = paused.
int lvPumpEvents(void);
}

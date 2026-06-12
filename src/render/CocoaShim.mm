// Programmatic AppKit shim for the live view (see CocoaShim.h).

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

static bool gQuit = false;
static bool gPause = false;

@interface LVDelegate : NSObject <NSWindowDelegate>
@end
@implementation LVDelegate
- (void)windowWillClose:(NSNotification*)n {
    gQuit = true;
}
@end

extern "C" void* lvCreateWindow(int width, int height, const char* title,
                                void* mtlDevice) {
    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        if (app == nil) return nullptr;
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        [app finishLaunching];

        const NSRect rect = NSMakeRect(0, 0, width, height);
        NSWindow* win = [[NSWindow alloc]
            initWithContentRect:rect
                      styleMask:(NSWindowStyleMaskTitled |
                                 NSWindowStyleMaskClosable |
                                 NSWindowStyleMaskMiniaturizable)
                        backing:NSBackingStoreBuffered
                          defer:NO];
        if (win == nil) return nullptr;
        win.title = [NSString stringWithUTF8String:title];
        static LVDelegate* delegate = [LVDelegate new];
        win.delegate = delegate;
        win.releasedWhenClosed = NO;

        NSView* view = [[NSView alloc] initWithFrame:rect];
        view.wantsLayer = YES;
        CAMetalLayer* layer = [CAMetalLayer layer];
        layer.device = (__bridge id<MTLDevice>)mtlDevice;
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer.framebufferOnly = YES;
        const CGFloat scale = win.backingScaleFactor;
        layer.contentsScale = scale;
        layer.drawableSize = CGSizeMake(width * scale, height * scale);
        view.layer = layer;
        win.contentView = view;

        [win center];
        [win makeKeyAndOrderFront:nil];
        [app activateIgnoringOtherApps:YES];
        return (__bridge_retained void*)layer; // view keeps it alive too
    }
}

extern "C" int lvPumpEvents(void) {
    @autoreleasepool {
        NSEvent* e;
        while ((e = [NSApp nextEventMatchingMask:NSEventMaskAny
                                       untilDate:[NSDate distantPast]
                                          inMode:NSDefaultRunLoopMode
                                         dequeue:YES]) != nil) {
            if (e.type == NSEventTypeKeyDown) {
                NSString* c = e.charactersIgnoringModifiers;
                if ([c isEqualToString:@" "]) {
                    gPause = !gPause;
                    continue;
                }
                if ([c isEqualToString:@"q"]) {
                    gQuit = true;
                    continue;
                }
            }
            [NSApp sendEvent:e];
        }
    }
    return gQuit ? 0 : (gPause ? 2 : 1);
}

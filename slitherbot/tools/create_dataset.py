import time
import mss

def take_screenshots(num_screenshots, output_dir="slitherbot/screenshots", start_index=0):
    """
    Capture screenshots for dataset creation.
    
    Args:
        num_screenshots (int): Number of screenshots to capture
        output_dir (str): Directory to save screenshots (default: "screenshots")
        start_index (int): Starting index for screenshot numbering (default: 0)
    """
    screenshot_count = 0
    
    with mss.mss() as sct:
        monitor_number = 1
        mon = sct.monitors[monitor_number]

        # Define capture region (adjust as needed)
        monitor = {
            'top': mon['top'] + 120, 
            'left': mon['left'], 
            'width': mon['width'], 
            'height': int(mon['height'] * 0.84), 
            'mon': monitor_number
        }

        print(f"Starting screenshot capture in 3 seconds...")
        time.sleep(3)
        
        while screenshot_count < num_screenshots:
            sct_img = sct.grab(monitor)
            output_path = f"{output_dir}/img{screenshot_count + start_index}.png"
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output_path)
            
            print(f"Screenshot {screenshot_count + 1}/{num_screenshots} saved: {output_path}")
            screenshot_count += 1
            time.sleep(0.1)
        
        print(f"\nCompleted! {num_screenshots} screenshots saved to '{output_dir}' directory.")

if __name__ == '__main__':
    try:
        num_screenshots = int(input("How many screenshots do you want to capture? "))
        if num_screenshots <= 0:
            raise ValueError("Number must be positive")
    except (TypeError, ValueError) as e:
        print("Error: Please enter a valid positive integer.")
        exit(1)

    start_index = 0
    try:
        start_idx_input = input("Starting index (press Enter for 0): ").strip()
        if start_idx_input:
            start_index = int(start_idx_input)
    except ValueError:
        print("Invalid index, using 0")
        start_index = 0

    input("\nPress Enter to start capturing...")
    take_screenshots(num_screenshots, start_index=start_index)
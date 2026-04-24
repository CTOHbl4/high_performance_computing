import slangpy
import pathlib
from scene import SceneCreator
import numpy as np
import cv2
import os
import glob

N = 128
M = 14


class InteractiveRenderer:
    def __init__(self, values_folder='floats', errors_folder='errors', rel_errors_folder='rel_errors', maxValue=1.0, maxError=1.0, maxRelError=1.0, skip_frames=1):
        self.device = slangpy.create_device(include_paths=[
            pathlib.Path(__file__).parent.absolute(),
        ])
        tracer = self.device.load_program(module_name="tracer_diff.slang", 
                                        entry_point_names=["pathsTrace"])
        self.tracer_k = self.device.create_compute_kernel(program=tracer)

        self.values_folder = values_folder
        self.values_files = self._get_data_files(values_folder)
        self.maxValue = maxValue
        self.errors_folder = errors_folder
        self.errors_files = self._get_data_files(errors_folder)
        self.maxError = maxError
        self.rel_errors_folder = rel_errors_folder
        self.rel_errors_files = self._get_data_files(rel_errors_folder)
        self.maxRelError = maxRelError
        
        if not self.values_files or not self.errors_files or not self.rel_errors_files:
            raise FileNotFoundError(f"No .bin files found in folders/")

        self.m = M
        self.skip_frames = skip_frames

        self.camera = {
            'pos': np.array([3.5, 1.0, 3.5]),
            'look_at': np.array([0.0, 0.0, 0.0]),
            'up': np.array([0.0, 1.0, 0.0])
        }

        self.scene = SceneCreator()(self.camera, N, self.m, self.device, self.tracer_k)
        self.window_name = "3D Volume Renderer"
        self.current_file_index = 0
        self.auto_advance = True
        self.frame_delay = 10
        
    def _get_data_files(self, folder):
        pattern = os.path.join(folder, folder + "*.bin")
        files = glob.glob(pattern)

        def extract_number(filename):
            base = os.path.basename(filename)
            digits = ''.join(filter(str.isdigit, base))
            return int(digits) if digits else 0
        
        return sorted(files, key=extract_number)
    
    def load_next_file(self, data_files, maxValue, var_radius=True):
        file_path = data_files[self.current_file_index]
        data = np.fromfile(file_path, dtype=np.float32)
        self.scene.update_function_values_and_max(data / maxValue, var_radius)
        return data.max(), data.min()
    
    def get_trio(self):
        results = []
        self.current_file_index = (self.current_file_index + self.skip_frames) % len(self.values_files)
        for data_files, maxValue, var_radius, title in zip((self.values_files, self.errors_files, self.rel_errors_files),
                                              (self.maxValue, self.maxError, self.maxRelError),
                                              (True, True, False),
                                              ('Values', 'Errors', 'Relative errors')):
            mx, mn = self.load_next_file(data_files, maxValue, var_radius)
            results.append(self.scene.make_pretty_frame(self.scene.forward(),
                                                        self.current_file_index,
                                                        len(self.values_files),
                                                        mx, mn, maxValue, title))
        err = np.concatenate([results[1], results[2]], axis=0)
        values = cv2.resize(results[0], (2 * results[0].shape[1], 2 * results[0].shape[0]), interpolation=cv2.INTER_NEAREST)
        return np.concatenate([values, err], axis=1)

    def rotate_camera_horizontal(self, angle_degrees):
        angle_rad = np.radians(angle_degrees)
        pos = self.camera['pos']

        new_x = pos[0] * np.cos(angle_rad) - pos[2] * np.sin(angle_rad)
        new_z = pos[0] * np.sin(angle_rad) + pos[2] * np.cos(angle_rad)
        
        self.camera['pos'] = np.array([new_x, pos[1], new_z])
        self.scene.update_camera(self.camera)
    
    def rotate_camera_vertical(self, angle_degrees):
        angle_rad = np.radians(angle_degrees)
        pos = self.camera['pos']

        radius = np.linalg.norm(pos)
        current_elevation = np.arcsin(pos[1] / radius)

        new_elevation = current_elevation + angle_rad
        max_elevation = np.pi / 2 - 0.1
        new_elevation = np.clip(new_elevation, -max_elevation, max_elevation)

        horizontal_distance = radius * np.cos(new_elevation)
        new_y = radius * np.sin(new_elevation)

        horizontal_dir = np.array([pos[0], 0, pos[2]])
        horizontal_dir = horizontal_dir / np.linalg.norm(horizontal_dir)
        
        new_pos = horizontal_dir * horizontal_distance
        new_pos[1] = new_y
        
        self.camera['pos'] = new_pos
        self.scene.update_camera(self.camera)
    
    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 'space' to toggle auto-advance")
        print("- Press 'a' to rotate left")
        print("- Press 'd' to rotate right")
        print("- Press 'w' to rotate up")
        print("- Press 's' to rotate down")
        print(f"- Auto-advance: {'ON' if self.auto_advance else 'OFF'}")
        
        rotation_speed = 5
        moving_forward_speed = 0.95
        moving_backward_speed = 1/moving_forward_speed
        
        try:
            while True:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user")
                    break

                if self.auto_advance:
                    frame = self.get_trio()
                cv2.imshow(self.window_name, frame)
                if self.auto_advance:
                    key = cv2.waitKey(self.frame_delay) & 0xFF
                else:
                    key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.auto_advance = not self.auto_advance
                    print(f"Auto-advance: {'ON' if self.auto_advance else 'OFF'}")
                elif key == ord('a'):
                    self.rotate_camera_horizontal(rotation_speed)
                    print(f"Rotated left, position: {self.camera['pos']}")
                elif key == ord('d'):
                    self.rotate_camera_horizontal(-rotation_speed)
                    print(f"Rotated right, position: {self.camera['pos']}")
                elif key == ord('w'):
                    self.rotate_camera_vertical(rotation_speed)
                    print(f"Rotated up, position: {self.camera['pos']}")
                elif key == ord('s'):
                    self.rotate_camera_vertical(-rotation_speed)
                    print(f"Rotated down, position: {self.camera['pos']}")
                elif key == ord('f'):
                    self.camera['pos'] *= moving_forward_speed
                    self.scene.update_camera(self.camera)
                    print(f"Moved forward, position: {self.camera['pos']}")
                elif key == ord('b'):
                    self.camera['pos'] *= moving_backward_speed
                    self.scene.update_camera(self.camera)
                    print(f"Moved backward, position: {self.camera['pos']}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cv2.destroyAllWindows()
            self.device.close()
            print("Cleanup completed")


def form_rel_errors(floats_dir='floats', errors_dir='errors', threshold=1e-3):
    max_value = -1.0
    max_error = 0.0
    max_rel_error = 0.0
    
    for file_value, file_error in zip(sorted(os.listdir(floats_dir)), 
                                       sorted(os.listdir(errors_dir))):
        values = np.fromfile(os.path.join(floats_dir, file_value), 
                            dtype=np.float32)
        errors = np.fromfile(os.path.join(errors_dir, file_error), 
                            dtype=np.float32)
        rel_errors = np.zeros_like(errors)
        max_value = max(np.max(values), max_value)
        max_error = max(np.max(errors), max_error)
        mask = np.abs(values) > threshold
        number = file_value[len(floats_dir):file_value.find('.')]
        if np.any(mask):
            rel_errors[mask] = np.max(errors[mask] / 
                                      np.abs(values[mask]))
            max_rel_error = max(np.max(rel_errors), max_rel_error)
        
        rel_errors.tofile(f'rel_errors/rel_errors{number}.bin')


def find_max_values_and_store_short_data(floats_dir='floats', errors_dir='errors', rel_errors_dir='rel_errors', orig_dir_prefix='full_', m=M, threshold=1e-3):
    max_value = -1.0
    max_error = 0.0
    max_rel_error = 0.0

    orig_floats_dir = orig_dir_prefix + floats_dir
    orig_errors_dir = orig_dir_prefix + errors_dir
    orig_rel_errors_dir = orig_dir_prefix + rel_errors_dir
    
    for file_value, file_error, file_rel_error in zip(sorted(os.listdir(orig_floats_dir)), 
                                       sorted(os.listdir(orig_errors_dir)), sorted(os.listdir(orig_rel_errors_dir))):
        values = np.fromfile(os.path.join(orig_floats_dir, file_value), 
                            dtype=np.float32).reshape((N, N, N))[::m, ::m, ::m]
        errors = np.fromfile(os.path.join(orig_errors_dir, file_error), 
                            dtype=np.float32).reshape((N, N, N))[::m, ::m, ::m]
        rel_errors = np.fromfile(os.path.join(orig_rel_errors_dir, file_rel_error), 
                            dtype=np.float32).reshape((N, N, N))[::m, ::m, ::m]
        max_value = max(np.max(values), max_value)
        max_error = max(np.max(errors), max_error)
        max_rel_error = max(np.max(rel_errors), max_rel_error)
        values.tofile(floats_dir + '/' + file_value)
        errors.tofile(errors_dir + '/' + file_error)
        rel_errors.tofile(rel_errors_dir + '/' + file_rel_error)
    
    print(f"Max value: {max_value}")
    print(f"Max absolute error: {max_error}")
    print(f"Max relative error (for |value|>{threshold}): {max_rel_error}")


if __name__ == '__main__':
    # form_rel_errors()
    # find_max_values_and_store_short_data()
    max_value = 1.0
    max_error = 0.000372
    max_rel_error = 1.0
    threshold = 1e-3
    renderer = InteractiveRenderer(maxValue=max_value, maxError=max_error, maxRelError=max_rel_error, skip_frames=5)
    renderer.run()

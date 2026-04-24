import slangpy
import numpy as np
import cv2

H = 400
W = 400

def fill_empty(lst: list):
    if not lst:
        lst.append(0)


class SceneCreator:

    class Scene:
        def float32_buffer(self, array):
            if isinstance(array, list):
                fill_empty(array)
            return self.device.create_buffer(format=slangpy.Format.r32_float, data=np.array(array, dtype=np.float32).flatten())

        def uint32_buffer(self, array):
            if isinstance(array, list):
                fill_empty(array)
            return self.device.create_buffer(format=slangpy.Format.r32_uint, data=np.array(array, dtype=np.uint32).flatten())

        def readonly_2dtexture(self, array, width, height):
            return self.device.create_texture(
                width=width,
                height=height,
                type=slangpy.TextureType.texture_2d,
                format=slangpy.Format.r32_float,
                usage=slangpy.TextureUsage.shader_resource,
                data=np.array(array, dtype=np.float32).flatten()
            )

        def fullaccess_2dtexture(self, array, width, height):
            return self.device.create_texture(
                width=width,
                height=height,
                type=slangpy.TextureType.texture_2d,
                format=slangpy.Format.r32_float,
                usage=slangpy.TextureUsage.shader_resource | slangpy.TextureUsage.unordered_access,
                data=np.array(array, dtype=np.float32).flatten()
            )

        def _tracer_dispatch(self):
            self.tracer_k.dispatch(thread_count=(W, H, 1),
                vars={
                    "size": [W, H],
                    "cameras": self.cameras,
                    "spheresNum": self.spheres_num,
                    "radiuses": self.radiuses,
                    "spheres": self.spheres,
                    "functionValues": self.function_values,
                    "output": self.output
                }
            )

        def forward(self):
            self._tracer_dispatch()
            return (np.clip(self.output.to_numpy().reshape((H, W, 3)), 0, 1) * 255).astype(np.uint8)

        def update_function_values_and_max(self, array, var_radius=True):
            self.function_values.copy_from_numpy(array)
            if var_radius:
                self.radiuses.copy_from_numpy(self.radius * (0.1 + np.abs(array)))
            else:
                self.radiuses.copy_from_numpy((self.radius / 2.0) * np.ones_like(array))

        def update_camera(self, camera):
            camera_params = SceneCreator._get_camera(camera, W, H)
            self.cameras.copy_from_numpy(camera_params)
        
        def make_pretty_frame(self, frame, frame_num, total_frames, top_value, bottom_value, maxValue, title, width=H//8):
            image = np.ones((H, width, 3), dtype=np.uint8) * int(0.9 * 255)
            gradient_height = H-100
            start_x = (H - gradient_height) // 2
            values = np.linspace(top_value/maxValue, bottom_value/maxValue, gradient_height)
            r = 0.5 + values / 2.0
            g = 0.5 - values / 2.0
            b = 1.0 - np.abs(values)
            colors = np.stack([b, g, r], axis=1)
            image[start_x:-start_x, 0:width] = colors[:, np.newaxis, :] * 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            thickness = 1
            top_text = f"{top_value:.4f}"
            bottom_text = f"{bottom_value:.4f}"
            top_x = (width - 37) // 2
            cv2.putText(image, top_text, (top_x, 15), 
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            bottom_x = (width - 47) // 2
            cv2.putText(image, bottom_text, (bottom_x, H - 10), 
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.putText(frame, f"time step / total steps in period: {frame_num}/{total_frames}", (10, 30), 
                        font, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, title, (H//3, 60),
                        font, 0.5, (0, 0, 0), 2)
            return np.concatenate([frame, image], axis=1)

    @staticmethod
    def _get_geometry(N, m=1):
        coords = np.linspace(-1, 1, N, endpoint=False)
        x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
        centers = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        sampled_centers = centers.reshape(N, N, N, 3)[::m, ::m, ::m].reshape(-1, 3)
        spheres = sampled_centers.flatten().astype(np.float32)
        return spheres

    @staticmethod
    def _get_camera(camera, out_w, out_h):
        cameras = np.zeros(14, np.float32)

        camera_pos = camera['pos']
        look_at = camera['look_at']
        up = camera['up']

        w = look_at - camera_pos
        w = w / np.linalg.norm(w)
        uvec = np.cross(w, up)
        uvec = uvec / np.linalg.norm(uvec)
        vvec = -np.cross(uvec, w)

        aspect = out_w / out_h
        cameras[:3] = camera_pos
        cameras[3:6] = w
        cameras[6:9] = uvec
        cameras[9:12] = vvec
        cameras[12] = np.tan(np.pi/8.0)
        cameras[13] = aspect
        return cameras

    def _get_scene_object(self, device, spheres, cameras, N, m):
        scene = SceneCreator.Scene()
        scene.device = device

        scene.spheres_num = len(spheres) // 3
        scene.spheres = scene.float32_buffer(spheres)
        scene.cameras = scene.float32_buffer(cameras)
        shape = np.zeros((N, N, N))[::m, ::m, ::m]
        scene.function_values = scene.float32_buffer(shape)
        scene.radiuses = scene.float32_buffer(shape)

        scene.output = scene.fullaccess_2dtexture(np.zeros(H * W * 3, dtype=np.float32), W * 3, H)
        return scene

    def main(self, camera, N, m, device, tracer_k):
        camera_params = self._get_camera(camera, W, H)
        spheres = self._get_geometry(N, m)
        
        scene = self._get_scene_object(device, spheres, camera_params, N, m)
        scene.device = device
        scene.tracer_k = tracer_k

        scene.radius = m / 2 / N

        return scene

    def __call__(self, camera, N, m, device, tracer_k):
        return self.main(camera, N, m, device, tracer_k)
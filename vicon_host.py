from __future__ import print_function
from vicon_dssdk import ViconDataStream
import argparse
import math
import socket
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEST_IP   = "164.11.72.61"   # Target IVP4 Adress Should Be Entered Here
DEST_PORT = 5005

SUBJECT = "Wand"          # the solved subject name
SEGMENT = "root"         

SEND_HZ = 60.0                
SEND_DT = 1.0 / SEND_HZ

# One Euro filter - tune live against the arm:
#   MIN_CUTOFF down = smoother but laggier
#   BETA       up   = more responsive to fast moves
MIN_CUTOFF = 1.0
BETA       = 0.02
D_CUTOFF   = 1.0

# ---------------------------------------------------------------------------


class OneEuro:
    """One Euro filter for a single scalar. One instance per axis."""
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def send(sock, dest, x, y, z, a, b, c, w):
    msg = f"{x:.3f},{y:.3f},{z:.3f},{a:.3f},{b:.3f},{c:.3f},{w:.3f}"
    sock.sendto(msg.encode("utf-8"), dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', nargs='?', default="localhost:801",
                        help="Vicon host, server:port")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (DEST_IP, DEST_PORT)

    fx, fy, fz = (OneEuro(MIN_CUTOFF, BETA, D_CUTOFF) for _ in range(3))
    last_valid = None   # last good (x, y, z) sent

    client = ViconDataStream.Client()
    try:
        client.Connect(args.host)
        print('Version', client.GetVersion())
        print(f"Streaming {SUBJECT}/{SEGMENT} -> udp {DEST_IP}:{DEST_PORT}")
        client.SetBufferSize(1)
        client.EnableSegmentData()
        client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)
        client.SetAxisMapping(ViconDataStream.Client.AxisMapping.EForward,
                              ViconDataStream.Client.AxisMapping.ELeft,
                              ViconDataStream.Client.AxisMapping.EUp)

        while True:
            client.GetFrame()
            t = time.time()

            (x, y, z), occluded = client.GetSegmentGlobalTranslation(SUBJECT, SEGMENT)
            (a, b, c, w), occluded = client.GetSegmentGlobalRotationQuaternion(SUBJECT, SEGMENT)

            # Don't trust the flag alone; also catch the exact-zero no-data case.
            bad = occluded or (x == 0.0 and y == 0.0 and z == 0.0)
            if bad:
                if last_valid is not None:
                    hx, hy, hz, ha, hb, hc, hw = last_valid
                    send(sock, dest, hx, hy, hz, ha, hb, hc, hw)   # held, not live
                time.sleep(SEND_DT)
                continue

            #xf, yf, zf = fx(x, t), fy(y, t), fz(z, t)
            last_valid = (x, y, z, a, b, c, w)

            send(sock, dest, x, y, z, a, b, c, w)

            time.sleep(SEND_DT)

    except ViconDataStream.DataStreamException as e:
        print('Handled data stream error', e)
    finally:
        client.Disconnect()
        sock.close()


if __name__ == "__main__":
    main()
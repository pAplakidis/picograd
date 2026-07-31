import argparse
import functools
import http.server
import os

from picograd.viz import trace_path


def make_handler(root, trace):
  class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
      if self.path.split("?", 1)[0] == "/traces/latest.json":
        if not os.path.exists(trace):
          self.send_error(404, f"Trace not found: {trace}")
          return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(os.path.getsize(trace)))
        self.end_headers()
        with open(trace, "rb") as f:
          self.copyfile(f, self.wfile)
        return
      return super().do_GET()
  return functools.partial(Handler, directory=root)


def main(argv=None):
  parser = argparse.ArgumentParser(description="Serve Picograd viz viewer")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", default=8000, type=int)
  parser.add_argument("--trace", default=trace_path())
  args = parser.parse_args(argv)
  root = os.path.dirname(__file__)
  handler = make_handler(root, args.trace)
  server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
  print(f"Serving Picograd viz at http://{args.host}:{args.port}/index.html")
  print(f"Trace: {args.trace}")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()


if __name__ == "__main__":
  main()

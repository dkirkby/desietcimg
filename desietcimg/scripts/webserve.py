import argparse
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import TCPServer


class LoggingHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        logging.info(format % args)


def webserve():
    parser = argparse.ArgumentParser(
        description='Serve files from the current directory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('-p', '--port', type=int, default=2020,
        help='port number to use')
    parser.add_argument('--logpath', type=str, metavar='PATH',
        help='Path where logging output should be written')
    args = parser.parse_args()

    # Configure logging.
    logging.basicConfig(filename=args.logpath, level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    with TCPServer(('', args.port), LoggingHandler) as httpd:
        logging.info('serving at port {0}'.format(args.port))
        httpd.serve_forever()

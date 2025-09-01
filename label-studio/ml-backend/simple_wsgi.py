"""
Simple WSGI application entry point.
"""

from simple_model import app

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 9090))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug)

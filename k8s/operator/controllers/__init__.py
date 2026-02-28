"""AumOS operator controllers package.

Each module in this package registers kopf handlers for a specific CRD.
They are imported by main.py at startup so their @kopf.on.* decorators
take effect before the event loop begins.
"""

========
ApsimX
========

Requirements
------------

- .NET 8.0 SDK library
- Gtk3 and GtkSourceView

Installation from source
------------------------

The basic steps for installing ApsimX are:

#. Install the depenedencies above
#. Clone the ApsimX repository from `here <https://github.com/APSIMInitiative/ApsimX>`_::

    git clone https://github.com/APSIMInitiative/ApsimX.git

#. Build the ApsimX.sln solution file::

    dotnet build path/to/ApsimX/ApsimX.sln

#. Point simulatr at the installation::

    python -m simulatr config directories apsimx path/to/ApsimX


See instructions `here <https://docs.apsim.info/docs/development/compile>`_ if you encounter errors when installing ApsimX.



# Introduction #

Checkout from SVN [Source](http://code.google.com/p/marioai/source/checkout) or download a [zip](http://www.marioai.com/marioai-benchmark/download/MarioAI.zip) package.

# Details #

Look for the specific project files in trunk. Project files for IntelliJ IDEA, Eclipse Galileo and Netbeans are provided.
Or you might also checkout from SVN using the above mentioned IDEs. Use SVN version 1.6 everywhere.

> ## IntelliJ IDEA**([JetBrains](http://www.jetbrains.com/idea/download/index.html) released a free community edition!) ##
    1. Version Control
    1. Checkout from Version Control.
    1. Subversion.
    1. http://marioai.googlecode.com/svn/trunk/ or http://marioai.googlecode.com/svn/ if you would like to checkout the entire project with branches, documentation and target folder, etc.
    1. Create a project from source in _Your\_Just\_Created\_Folder_? Yes
    1.**Next>**(if you satisfied with suggested project name and path; otherwise, change it!)
    1. Mark All if you want IDEA to recognize all java files in source directories.
    1.**Next>**1. Mark `jdom.jar` and press**Next>**1. Select _Trunk_ or _Src_ in**Modules**column. In the right column**Module Dependencies**you should see `jdom.jar`
    1.**Next >**1. Exclude if any facets in Python src found
    1.**Finish**!
    1. Go to**Preferences**>**Compiler**1. Add here `;*.lvl;*.dat` (this makes IDEA to copy these files within the compiled binary class-files; so the IDE could find it while running)
    1.**OK

> ## Eclipse Galileo for Python parts of the project only**##
    1. File -> New -> Other
    1. SVN -> Checkout projects from SVN
    1. Create a new repository location http://marioai.googlecode.com/svn/trunk
    1.**Next >**1. src -> python -> competition
    1.**Next >**1. Select "Check out as a project configured using the New Project Wizard"
    1.**Next >**(You will see a project creation wizard)
    1. _Pydev_ -> _Pydev_ Project ([Pydev](http://pydev.org) plugin for Eclipse should be installed)
    1. Give name to a project, say "MarioAICompetition"
    1. uncheck "Create default 'src'..."
    1.**Finish

> ## Eclipse Galileo for Java part and entire source**##
    1. File -> New -> Other
    1. SVN -> Checkout projects from SVN
    1. Create a new repository location http://marioai.googlecode.com/svn/trunk or use the existing one
    1.**Next >**1. Select "Check out as a project configured using the New Project Wizard"
    1.**Next >**(You will see a project creation wizard)
    1. Select "_Java_ _Project_"
    1. Name the project, say "MarioAI" and select a location
    1. Make root folder as source folder
    1. Configure inclusion and exclusion filters
    1. Press**Add**and put down '**.lvl', '**.dat'
    1.**Finish

> ## Netbeans IDE**##
    * Please, let  us know, if one need instructions for Netbeans (which are basically similar to the aboves)**

> Once your project had been set up, you can go to [Overview](Overview.md), run the becnhmark, [Play](Play.md) and start programming your [Agent](Agent.md)!

> 
---

> This page is finished unless any request will appear.
Comments and requests are welcome!
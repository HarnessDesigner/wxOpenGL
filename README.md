# wxOpenGL
wxPython OpenGL framework

This project aims to make it easier to use OpenGL from Python using the wsPython 
cross platform GUI framework.

***WARNING***: Still in development

<br/>

Goals
_____

* ✅️ Child Window: OpenGL window as a child window so it can embedded into
                an application.
* ✅️️ OpenGL Context handling: Allows multiple OpenGL instances.
* ✅️ OpenGL Material handling: Wrappers that make dealing with OpenGL materials
                            easier to do.
* ⬜️ Movement/positioning: Built in mouse control that makes sense. 
* ✅️ Object containers: Objects are the things being rendered. This can be an entire 
                     scene or a single piece where groups would make up a scene.
* ✅️ Object selection: Ability to select an object

<br/>

***Supported 3d model formats***
________________________________
* 3D GameStudio Model ( .mdl )
* 3D GameStudio Terrain ( .hmp )
* 3D Manufacturing Format (.3mf)
* 3D Studio Max 3DS (.3ds)
* 3D Studio Max ASE (.ase)
* AC3D (.ac, .ac3d)
* AutoCAD DXF (.dxf)
* Autodesk DXF (.dxf)
* Biovision BVH (.bvh)
* Blender (.blend)
* CharacterStudio Motion ( .csm )
* Collada (.dae, .xml)
* DirectX X (.x)
* Doom 3 (.md5mesh, .md5anim, .md5camera)
* FBX-Format, as ASCII and binary (.fbx)
* glTF (.glTF)
* glTF2.0 (.glTF)
* IFC-STEP (.ifc)
* Initial Graphics Exchange Specification (.igs, .iges)
* Irrlicht Mesh (.irrmesh, .xml)
* Irrlicht Scene (.irr, .xml)
* KHR_lights_punctual ( 5.0 )
* KHR_materials_pbrSpecularGlossiness ( 5.0 )
* KHR_materials_unlit ( 5.0 )
* KHR_texture_transform ( 5.1 under test )
* LightWave Model ( .lwo )
* LightWave Scene ( .lws )
* Milkshape 3D ( .ms3d )
* Modo Model ( .lxo )
* Neutral File Format (.nff)
* Object File Format ( .off )
* Ogre ( .mesh.xml, .skeleton.xml, .material )
* OpenGEX-Fomat (.ogex)
* Quake I (.mdl)
* Quake II (.md2)
* Quake III (.md3)
* Quake 3 BSP (.pk3)
* Quick3D (.q3o, .q3s)
* Raw Triangles (.raw)
* RtCW (.mdc)
* Sense8 WorldToolkit (.nff)
* Standard for the Exchange of Product Data (.stp, .step)
* Polygon File Format (.ply)
* Stanford Triangle Format ( .ply )
* Stereolithography (.stl)
* Terragen Terrain ( .ter )
* TrueSpace (.cob, .scn)
* Valve Model (.smd, .vta)
* Virtual Reality Modeling Language (.wrl)
* Wavefront Object (.obj)
* XGL-3D-Format (.xgl)
* AMF: Additive manufacturing file format (.amf)
* ASSBIN: Allegorithmic Substance Painter Scene within Document (.assbin)
* B3D: OpenBVE 3D (.b3d)
* IQM: Inter-Quake Model (.iqm)
* NDO: 3D Low-polygon Modeler (.ndo)
* Q3D: Quest3D (.q3d)
* SIB: Silo Model Format (.sib)
* X3D: Extensible 3D (.x3d, .x3db, x3dz, x3dbz)
* GLB: (.glb)
* MMD: MikuMikuDance Format (.pmd, .pmx)



***How the control works...***
______________________________

* Left mouse click and drag is look up/down and left/right
* Right mouse click and drag is move up/down  and sidestep left/right. 
  The focal point moves with the camera.
* Middle click and drag has a fixed focal point and the camera move around that point.
* Left click on an object will select it. The object will become translucent and glow blue.
  To deselect the object either you need to left click on the same object or left click 
  on a different object. You have the ability to move around with the object staying selected. 
* Mouse wheel walks forwards and backwards.
* Left click and drag an object when it is selected will move the object. I got crafty
  with the mechanics opn this because almost every 3D CAD stiyle program I have used has
  always had an object move erraticaly when moving using the mouse. So what I did to solve 
  that problem is an axis lock. The axis lock occurs when you first move the mouse.
  Make sure to move the mouse in the direction you want to move the object and it will then
  lock to moving on that axis. The lock gets released only when you release the mouse button.
  the object will stay selected after releasing the mouse button. I set it up like this so 
  direction changes are simple to do.
* Right click and drag on a selected object will rotate the object. The control function in 
  the same manner with the rotation being locked to a specific axis once you move the mouse.
  > NOTE: This feature I am still working on and it does not function properly at the time being.



***Extra features***
____________________

There is a mechanism that I built in that holds config settings. This mechanism
stores the config settings in a sqlite3 database. By default the database is created
in memory so the setting are not persistanct between reloads of the library. You can provide
a path to save the settings to file so it will be persistant. Currently the config settings 
are shared if you have more than one instance of the library running. I will be making some
changes to the settings to allow for separate settings for each instance that is running.

Here are the config settings...


* ground_height (default: `0.0`): Sets where the "floor" is located.
                                This is the bottom limit of object positioning 
                                and also how low the camera is able to move.

* eye_height (default `10.0`): Starting height of the camera. This setting is not being used yet.

* reflections (default `False`): Show reflections on the floor. `grid.render` must also be set 
                                 to `True`. This is not being used yet.

* reflection_strength (default `50.0`): Range is 0.0 - 100.0. This is how strong the reflection is.
                                      Not being used yet.   

* grid: Grid settings
  * render (defualt `True`): Turn on and off rendering the grid floor. The impact to performance for 
                             rendering the floor is very small. There is a startup time penatly of 
                             about 7ms. The floor gets build and then stored into the memory on the 
                             video card which makes it super fast to render when a refresh occurs. 
                             I am talking 0.13ms fast.  

  * size (default `1000`): This is how far the floor goes in all directions. The clipping plane
                           for far is hard coded at 1000. so the floor is defulted to that clipping 
                           distance. The floow distance is measure as 1000 from `(0, ground_height, 0)`
                           that means the total flore area has a size of `(2000, 2000)`.

  * step (default `50`): This is the size of the checkerboard. Like the `size` this is a 1/2 value so 
                         setting it to 50 produces squares that have a size of `(100, 100)` 

  * odd_color (default `[0.3, 0.3, 0.3, 0.8]`): Odd square color. This is a scalar value for 
                                                `[red, green, blue, alpha]`
                                                each component has a range of `0.0` to `1.0`

  * even_color (default `[0.8, 0.8, 0.8, 0.8]`): Even square colors. Works the same as the `odd_color`
                                                 setting.

* virtual_canvas: One of the things I did not care for was resizing the window and having
                  the objects in view change viewable size. I wanted to keep the appearance the same
                  and increase the field of view when making the window larger or smaller.
                  This was a but tricky to do because I had to wrap the OpenGL window with a `wx.Panel`
                  and the gl window is not coupled to a sizer. The gl window is always the same size
                  and the panel that holds the gl window is what clips the view.

  * width (default `1920`): Sets the width of the gl window. If you need to change
                            this setting you will want to keep the aspect ratio of 16:9
                            between the width and height otherwise things can get distored. 

  * height (defualt `1080`): Sets the height, same rules apply that are outlined above. 

* keyboard_settings: These settings control how fast key repeats fired and how fast the speed ramps up.

  * max_speed_factor (default `10.0`): This the max the factorcan get. The facor is a number that starts
                                       at `start_speed_factor` and with each loop of the thread if the 
                                       key is being held down the factor is increased by 
                                       `speed_factor_increment` until it hits the `max_speed_factor`

  * speed_factor_increment (default `0.1`): read above

  * start_speed_factor (default `1.0`): read above

This next group of config settings controls what button or key does what
and there is a sensitivity adjustment as well.

the the container name describes the movement. what you will see inside the containers
are setting names that describe the movement for keypresses and there is a `mouse`
setting where you would use one of the following constants..

* CONFIG_MOUSE_NONE: No mouse button applied
* CONFIG_MOUSE_LEFT: Left mouse button
* CONFIG_MOUSE_MIDDLE: Middle mouse button
* CONFIG_MOUSE_RIGHT: Right bouse button
* CONFIG_MOUSE_AUX1: Aux1 mouse button
* CONFIG_MOUSE_AUX2: Aux2 mouse button
* CONFIG_MOUSE_WHEEL: Vertical scroll wheel

The next bunch of constants are modifiers to alter the behavior of the movement in relationship
to the direction of travel the mouse is moving in. These options get OR'ed `|` to the constants above

* CONFIG_MOUSE_REVERSE_X_AXIS: Flips the X axis to right mouse movement becomes left and left becomes right

* CONFIG_MOUSE_REVERSE_Y_AXIS: Flips the Y axis so up becomes down and down becomes up

* CONFIG_MOUSE_REVERSE_WHEEL_AXIS: Flips the mouse wheel rotation

* CONFIG_MOUSE_SWAP_AXIS: swaps the X and Y axis so up/down becomes left/right and left/right becomes up/down


Setting the keys to be used needs to be set as the decimal value of the character the key represents
as seen on an ASCII chart. You don't need to look at an ASCII chart because Python has a nice built in function
that will do that conversion for us. The function is `ord`. 

Here are the movement settings..

* rotate 
  * mouse: default `MOUSE_MIDDLE`
  * up_key: default `ord('w')`
  * down_key: default `ord('s')`
  * left_key: default `ord('a')`
  * right_key: default `ord('d')`
  * sensitivity: default `0.4`

* pan_tilt
  * mouse: default `MOUSE_LEFT`
  * up_key: default `ord('o')`
  * down_key: default `ord('l')`
  * left_key: default `ord('k')`
  * right_key: default `ord(';')`
  * sensitivity: default `0.2`

* truck_pedestal
  * mouse: default `MOUSE_RIGHT`
  * up_key: default `ord('8')`
  * down_key: default `ord('2')`
  * left_key: default `ord('4')`
  * right_key: default `ord('6')`
  * sensitivity: default `0.2`
  * speed default: `1.0`

* walk 
  * mouse: default `MOUSE_WHEEL | MOUSE_SWAP_AXIS`
  * forward_key: default `wx.WXK_UP`
  * backward_key: default `wx.WXK_DOWN`
  * left_key: default `wx.WXK_LEFT`
  * right_key: default `wx.WXK_RIGHT`
  * sensitivity: default `1.0`
  * speed: default `5.0`

* zoom
  * mouse: default `MOUSE_NONE`
  * in_key: default `wx.WXK_ADD`
  * out_key: default `wx.WXK_SUBTRACT`
  * sensitivity: default `5.0`

* reset
  * mouse: default `MOUSE_NONE`
  * key: default `wx.WXK_HOME`



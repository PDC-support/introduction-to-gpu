# How to use OpenSYCL

## On Lumi

Install hipSYCL (OpenSYCL) locally:
```
module load LUMI/22.08 partition/G EasyBuild-user
eb --robot /appl/lumi/LUMI-EasyBuild-contrib/easybuild/easyconfigs/h/hipSYCL/hipSYCL-0.9.4-cpeGNU-22.08.eb
```

Then, load the module:
```
module load LUMI/22.08 partition/G EasyBuild-user hipSYCL/0.9.4-cpeGNU-22.08
```

and compile with
```
make
```

## On Dardel

Load a hipSYCL (OpenSYCL) module
```
ml PDC/22.06
ml hipsycl/0.9.4-cpeGNU-22.06
```

and compile with
```
make
```

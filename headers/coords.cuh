#ifndef _DATA3D_H_
#define _DATA3D_H_

// Basic struct to hold the 3 components of the 3D cartesian system
// Used for both position and velocity
struct data3D {
    double x,y,z;
    // Default constructor, sets all elements to zero
    __host__ __device__ data3D() {
        this->x = 0;
        this->y = 0;
        this->z = 0;
    }
    // Constructor that takes in values to set
    __host__ __device__ data3D(double setX, double setY, double setZ) {
        this->x = setX;
        this->y = setY;
        this->z = setZ;
    }
};

std::ostream& operator<<(std::ostream& os, const data3D& object) {
    os << "X:" << object.x << "\n";
    os << "Y:" << object.y << "\n";
    os << "Z:" << object.z << "\n";
    return os;
}

#endif